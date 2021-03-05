
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy import fftpack
import concurrent.futures
from functools import partial


def single_trial(dt, gamma, N0, sigma, dW):
    ''' for computing a single trial of stochastic evolution for population
    Inputs:
    dt = time step in fs
    gamma = decay rate in 1/fs
    N0 = starting value of population
    sigma = std of brownian process units are in 1/fs**0.5
    dW = an array of brownian steps'''
    n = np.ones(len(dW)+1)*0
    n[0] = N0
    for h in range(len(dW)):
        n[h+1] = n[h] + 1* (-gamma*n[h]*dt + sigma*dW[h]) #replace the second term with appropreate SDE
        # if n[h+1]<=0:
        #    break
    return n


def parallel_finite_difference(N0=2, nsteps=10000, dt=0.1, trials=1000, gamma=0.012/0.6582, sigma=(0.0025)**0.5):
    '''
    parallelized version for computing the solution of an SDE using finite difference method
    this function computes the numeric intergral of a stochastic process through the first order approximation of a
     taylor expansion
    u((i+1)h) = u(ih) + h*u'(ih)

    N_0 is the initial value for k!=0 excitons
    trials are the number of times a simulation is repeated for the same starting value of N
    nsteps are the number of time steps
    dt is the size of time step in fs
    gamma is the decay rate in 1/fs
    sigma is the sqrt of variance in 1/fs ** 0.5
    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        t = np.linspace(0.0, nsteps * dt, nsteps + 1)
        dW = (dt**0.5) * random.normal(0, 1, (trials, nsteps))
        func = partial(single_trial, dt, gamma, N0, sigma)
        f = executor.map(func, dW)
        n = []
        for x in f:
            n.append(x)
        n = np.array(n)
        # plt.figure()
        # for k in range(10):
        #     plt.plot(t, n[k])
        # plt.plot(t, np.mean(n, 0), linewidth=3)
        #plt.show()

        return t, n


def single_spec(mu, ng,  w0, V0, t, phiN1):
    '''compute a single spectrum for a single trial of N. Used in parallelization
    Inputs:
    mu = dipole strength, unitless
    ng = ground state population, unitless
    w0 = oscialltion frequency in 1/fs
    V0 = interaction strength in 1/fs
    t = time array in fs
    phiN1 = numerical integral of a single trial of stochastic evolution in N of k!=0'''
    s1 = np.zeros((len(phiN1)), dtype=complex)  # container for the first order response
    hbar = 0.6582  # eV.fs
    c1 = 2 * (mu ** 2) / hbar
    for k in range(len(s1)):
        c2 = cmath.exp(1j*t[k] * (w0 + V0*ng) + 2j*V0*phiN1[k])
        c3 = (cmath.exp(-1j*V0*t[k]) - 1)*ng - 1
        c = c2 * c3
        s1[k] = -c1 * c.imag
    return s1


def first_order_spec(dt=0.1, nsteps=100000, gamma=0.012, sigma=0.0025 ** 0.5, N0=2, trials=1000, samples=100, parallel=True):
    '''main function for computing the first order spectrum through finite element solution of SDE.
    Inputs
    dt = time step in fs
    nsteps = number of time-steps,
    gamma = decay rate given in eV then converted to 1/fs inside function
    sigma = brownian std in 1/fs
    sigmaN0 = laser std or initial exciton std. NOTE: it is unit-less
    V0 = interaction fixed at (10 meV/1000hbar) in 1/fs
    N0 = mean of initial population for k!=0 excitons
    trials = number of paths to average for each starting value of N0
    samples = number of samples from the initial population of k!=0 excitons
    Returns an averaged spectrum for a single starting volue of the N(0)'''

    hbar = 0.6582  # eV.fs
    gamma = gamma /hbar  # per fs
    ng = 1  # the ground state population of k=0 excitons

    # if parallel:
    #     t, n = parallel_finite_difference(N0, nsteps, dt, trials, gamma, sigma)
    # else:
    #     t, n = finite_difference_method(n_0, nsteps, dt, trials, gamma, sigma)  # note n's first element is n0


    t, n = parallel_finite_difference(N0, nsteps, dt, trials, gamma, sigma)
    phi_n = np.zeros(n.shape)
    phi_n[:, 1:] = dt * np.cumsum(n[:, :nsteps], axis=-1) # computing the integral of N(t)
    # notice the first element of phi_n is skipped from the np.cumsum to avoid double counting of the starting value
    # print phi_n to confirm if the integration is performed properly without double counting.
    V0 = 0.010/hbar  # the interaction potential in 1/fs
    w0 = 2.35/hbar  # the frequency in 1/fs
    mu = 1 # dipole moment in some arb units.
    c1 = 2 * (mu ** 2) / hbar

    # if parallel:
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         func = partial(single_spec, hbar, mu, ng,  w0, V0, t)
    #         f = executor.map(func, phi_n)
    #         s1 = []
    #         for x in f:
    #             s1.append(x)
    #         s1 = np.array(s1)
    # else:
    #     s1 = np.zeros((trials, nsteps + 1), dtype=complex) #container for the first order response
    #     for m in range(trials):
    #         for k in range(nsteps + 1):
    #             c2 = cmath.exp((+1j / hbar) * (t[k] * (w0 + V0 * ng) + 2 * V0 * phi_n[m, k]))
    #             c3 = (cmath.exp(-1j*V0*t[k])-1)*ng - 1
    #             c = c2*c3
    #             s1[m, k] = -c1*c.imag

    with concurrent.futures.ProcessPoolExecutor() as executor:
        func = partial(single_spec, mu, ng, w0, V0, t)
        f = executor.map(func, phi_n)
        s1 = []
        for x in f:
            s1.append(x)
        s1 = np.array(s1) # a 2D array containing all the spectra associated with all trials

    s1_mean = np.mean(s1, 0)
    s1dummy = np.ones(nsteps + 1, dtype=complex)  # spectrum without broadening
    s1dummy2 = np.ones(nsteps + 1, dtype=complex)  # spectrum without broadening and without shift
    for k in range(nsteps + 1):
        c2 = cmath.exp(1j*t[k]*(w0 + V0*ng))
        c3 = (cmath.exp(-1j*V0*t[k])-1)*ng - 1
        c = c2*c3
        s1dummy[k] = -c1*c.imag
        s1dummy2[k] = c1*(cmath.exp(+1j*t[k]*w0)).imag
#################################################################3
    # plt.figure()
    # plt.plot(t, s1_mean)
    #plt.plot(t, s1dummy)
    #plt.plot(t, s1dummy2)
    #plt.show()

    spec_s1 = np.fft.fft(s1_mean) / len(t)
    spec_s1dummy = np.fft.fft(s1dummy) / len(t)
    spec_s1dummy2 = np.fft.fft(s1dummy2) / len(t)
    freq = np.fft.fftfreq(len(spec_s1), d=dt)
    energy = 2 * np.pi * hbar * freq

    limit = int(nsteps / 2)
    # print(limit)
    # plt.figure()
    # plt.plot(energy[:limit], np.abs(spec_s1[:limit]))
    # plt.plot(energy[:limit], np.abs(spec_s1dummy[:limit]))
    # plt.plot(energy[:limit], np.abs(spec_s1dummy2[:limit]))
    # plt.legend(['stochastic + interacting', 'interacting', 'original'])
    # plt.title('Spectrum of first order response')
    # plt.xlabel('Energy or Freq. (eV)')
    # plt.ylabel('S1(w)')
    #plt.show()

    return energy[:limit], np.abs(spec_s1[:limit])


def laserbroadening(dt=0.1, nsteps=100000, gamma=0.012, sigma=0.0025 ** 0.5, sigmaN0=0.125 ** 0.5, N0=2, trials=1000, samples=100):
    n0 = sigmaN0 * random.normal(0, 1, samples) + N0
    plt.figure()
    plt.hist(n0)

    spec = []
    energy = []
    for n in n0:
        energy, s = first_order_spec(dt, nsteps, gamma, sigma, n, trials)
        spec.append(s)


    spec = np.array(spec)
    print(spec.shape, 'the spec shape')
    s1_mean = np.mean(spec, 0)
    print(s1_mean.shape, 'the shape of mean')
    plt.figure()
    plt.plot(energy, spec[0])
    plt.plot(energy, spec[1])
    plt.plot(energy, s1_mean)
    plt.legend(['sample 1', 'sample2', 'mean'])
    plt.title('Spectrum of first order response')
    plt.xlabel('Energy or Freq. (eV)')
    plt.ylabel('S1(w)')
    # plt.show()

    return energy, s1_mean




def first_order_exact(dt=0.1, nsteps=100000, gamma=0.012, sigma=0.0025 ** 0.5, sigmaN0=0.125 ** 0.5, N0=2):
    '''Exact formulation of the first order response as calculated by Hao et. al.
    time is in fs, rate 1/fs, frequencies or energies in eV

    Inputs
    dt = time step,
    nsteps = number of time-steps,
    gamma = decay rate given in eV
    sigma = brownian std in 1/fs
    sigmaN0 = laser std or initial exciton std. NOTE: it is unit less
    V0 = interaction fixed at (10 meV/1000hbar) in 1/fs
    N0 = mean of initial population for k!=0 excitons
        Returns the spectrum and frequencies'''


    hbar = 0.6582  # eV.fs Planck's constant
    w = 2.35/hbar # excitation center frequency in 1/fs
    V0 = 0.01/hbar # interaction potential in 1/fs
    ng = 1  # the ground state population of k=0 excitons
    gamma = gamma/hbar  # 1/fs
    mu = 1 # dipole strength
    time = np.linspace(0,dt*nsteps,nsteps)
    s1 = [(-2*mu/hbar)*np.imag(((np.exp(-1j*V0*t)-1)*ng-1)*np.exp(1j*(w+V0*ng)*t)\
                                * np.exp(2j*V0*N0*(1-np.exp(-gamma*t))/gamma)\
                                * np.exp(-((V0*sigma)**2/gamma**3) * (2*gamma*t + 4*np.exp(-gamma*t) - np.exp(-2*gamma*t)-3)\
                                      - 2*((V0*sigmaN0/gamma)**2)*(1-np.exp(-gamma*t))**2))\
                                for t in time]
    S1 = np.fft.fft(s1) / len(time)
    freq = np.fft.fftfreq(len(S1), d=dt)
    energy = 2*np.pi*hbar*freq


    # plt.figure()
    # plt.plot(time, s1)
    # plt.title('First order response in time domain')
    # plt.xlabel('time (fs)')
    # plt.ylabel('s1')

    limit = int(nsteps/2)
    # print(limit)
    # plt.figure()
    # plt.plot(energy[:limit], np.abs(S1[:limit]))
    # plt.title('Spectrum of first order response')
    # plt.xlabel('Energy or Freq. (eV)')
    # plt.ylabel('S1(w)')
    # plt.show()

    return energy[:limit], np.abs(S1[:limit])
