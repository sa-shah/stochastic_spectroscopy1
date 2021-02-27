
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy import fftpack
import concurrent.futures
from functools import partial


def single_trial(dt, gamma, n_0, sigma, dW):
    '''for computing a single trial of stochastic evolution for population'''
    n = np.ones(len(dW)+1)*0
    n[0] = n_0
    for h in range(len(dW)):
        n[h+1] = n[h] + 1* (-gamma*n[h]*dt + sigma*dW[h])
        if n[h+1]<=0:
           break
    return n


def parallel_finite_difference(n_0=2, nsteps=10000, dt=0.1, trials=1000, gamma=0.15, sigma=(0.0025)**0.5):
    '''parallelized version for computing the solution of an SDE using finite difference method
    this function computes the numeric intergral of a stochastic process through the first order approximation of a
     taylor expansion
    u((i+1)h) = u(ih) + h*u'(ih)

    n_0 is the initial value
    trials are the number of times a simulation is repeated for the same starting value
    nsteps are the number of time steps
    dt is the size of time step
    gamma is the decay rate
    sigma is the sqrt of variance'''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        t = np.linspace(0.0, nsteps * dt, nsteps + 1)
        dW = (dt**0.5) * random.normal(0,1,(trials, nsteps))
        func = partial(single_trial, dt, gamma, n_0, sigma)
        f = executor.map(func,dW)
        n = []
        for x in f:
            n.append(x)
        n = np.array(n)
        plt.figure()
        for k in range(10):
            plt.plot(t, n[k])
        plt.plot(t, np.mean(n, 0), linewidth=3)
        #plt.show()

        return t,n


def finite_difference_method(n_0=2, nsteps=1000, dt=0.1, trials=100, gamma=0.15, sigma=(0.0025)**0.5):
    """ this function computes the numeric intergral of a stochastic process through the first order approximation of a
     taylor expansion
    u((i+1)h) = u(ih) + h*u'(ih)

    n_0 is the initial value
    trials are the number of times a simulation is repeated for the same starting value
    nsteps are the number of time steps
    dt is the size of time step
    gamma is the decay rate
    sigma is the sqrt of variance
    """
    dW = (dt ** 0.5) * random.normal(0, 1, (trials, nsteps))
    n = np.ones((trials, nsteps+1))*1
    n[:, 0] = n_0
    for m in range(trials):
        for h in range(nsteps):
            n_prime = (-gamma*n[m, h]*dt + sigma*dW[m, h])
            n[m, h+1] = n[m, h] + 1*n_prime
            #if n[m, h+1]<=0:
             #   n[m, h+1:] = np.zeros(nsteps-h)
             #   break

    t = np.linspace(0.0, nsteps * dt, nsteps + 1)
    #N_mean = n_0*np.exp(-gamma*t)
    plt.figure()
    for k in range(10):
        plt.plot(t, n[k])
    plt.plot(t, np.mean(n, 0), linewidth=3)
    #plt.plot(t,N_mean, linewidth=3)
    #plt.show()

    return t, n


def brownian(x0, n, dt, out=None, add_ini=True, c_sum=True):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)
    print(x0.shape + (n,))
    """For each element of x0, generate a sample of n numbers from a normal distribution """
    #r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))
    #r = random.normal(0, dt**0.5, x0.shape+(n,))
    r = (dt ** 0.5) * random.normal(0, 1, x0.shape + (n,))
    #print(r.shape)
    #plt.figure()
    #plt.plot(r[0][:])
    #plt.show()
    """size is the number of random values, scale is the std of distribution, and loc = location (mu)"""
    #plt.figure()
    #plt.hist(np.squeeze(r[0][:]))
    #plt.show()
    """ If `out` was not given, create an output array."""
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    if c_sum:
        np.cumsum(r, axis=-1, out=out)
    else:
        out = r

    # Add the initial condition.
    if add_ini:
        out += np.expand_dims(x0, axis=-1)

    return out


def MC_OperatorEvolution():
    '''old function not in use anymore'''
    trials = 1000  # number of trajectories to run
    dt = 0.01
    nsteps = 100000
    hbar = 0.6582  # eV.fs
    gamma = 15  # meV
    gamma = gamma/(hbar*1000) # per fs
    sigma = 0.0025 ** 0.5  # per fs
    n_0 = 2 # initial value of the k!=0 exciton population
    ng = 1 # the ground state population of k=0 excitons
    t, n = finite_difference_method(n_0, nsteps, dt, trials, gamma, sigma) #note n's first element is n0
    phi_n = np.zeros(n.shape)
    phi_n[:,1:] =dt*np.cumsum(n[:,:nsteps],axis=-1)
    plt.figure()
    for k in range(10):
        plt.plot(t, phi_n[k])
    plt.plot(t, np.mean(phi_n, 0), linewidth=3)
    plt.show()
    V0 = 0.010  # the interaction potential in eV
    w0 = 2.35  # the frequency in energy units eV
    hbar = 0.658 # eV.fs
    a = np.zeros((trials, nsteps + 1), dtype=complex)
    for m in range(trials):
        for k in range(nsteps+1):
            a[m, k] = cmath.exp((-1j/hbar)*(t[k]*(w0 + V0*ng) + 2*V0*phi_n[m,k]))
    a_mean = np.mean(a, 0)
    adummy = np.ones(nsteps + 1, dtype=complex) # operator a without broadening
    for k in range(nsteps + 1):
        adummy[k] = cmath.exp((-1j/hbar) * t[k] * (w0 + V0 * n_0 + 0))

    plt.figure()
    plt.plot(t, np.imag(adummy))
    plt.plot(t, np.imag(a_mean))
    plt.show()

    spec_a = fftpack.fftshift(fftpack.fft(a_mean))
    spec_adummy = fftpack.fftshift(fftpack.fft(adummy))
    freq = fftpack.fftshift(fftpack.fftfreq(t.shape[-1], dt)) * (2 * np.pi)

    plt.figure()
    plt.plot(freq, abs(spec_adummy))
    plt.plot(freq, abs(spec_a))
    plt.show()


def single_spec(hbar, mu, ng,  w0, V0, t, phiN1):
    '''compute a single spectrum for a single trial of N. Used in parallelization'''
    s1 = np.zeros((len(phiN1)), dtype=complex)  # container for the first order response
    c1 = 2 * (mu ** 2) / hbar
    for k in range(len(s1)):
        c2 = cmath.exp((+1j / hbar) * (t[k] * (w0 + V0 * ng) + 2 * V0 * phiN1[k]))
        c3 = (cmath.exp(-1j * V0 * t[k]) - 1) * ng - 1
        c = c2 * c3
        s1[k] = -c1 * c.imag
    return s1


def first_order_spec(parallel=True, n_0=2):
    '''main function for computing the first order spectrum through finite element solution of SDE.
    Returns an averaged spectrum for a single starting volue of the N(0)'''
    trials = 1000 # number of trajectories to run
    dt = 0.1
    nsteps = 100000
    hbar = 0.6582  # eV.fs
    gamma = 10  # meV
    gamma = gamma / (hbar * 1000)  # per fs
    sigma = 0.0025 ** 0.5  # per fs
    n_0 = 2  # initial value of the k!=0 exciton population
    ng = 1  # the ground state population of k=0 excitons
    if parallel:
        t,n = parallel_finite_difference(n_0, nsteps, dt, trials, gamma, sigma)
    else:
        t, n = finite_difference_method(n_0, nsteps, dt, trials, gamma, sigma)  # note n's first element is n0
    phi_n = np.zeros(n.shape)
    phi_n[:, 1:] = dt * np.cumsum(n[:, :nsteps], axis=-1)

    V0 = 0.010  # the interaction potential in eV
    w0 = 2.35  # the frequency in energy units eV
    hbar = 0.658  # eV.fs
    mu = 1 # dipole moment in some arb units.
    c1 = 2 * (mu ** 2) / hbar
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            func = partial(single_spec, hbar, mu, ng,  w0, V0, t)
            f = executor.map(func, phi_n)
            s1 = []
            for x in f:
                s1.append(x)
            s1 = np.array(s1)
    else:
        s1 = np.zeros((trials, nsteps + 1), dtype=complex) #container for the first order response
        for m in range(trials):
            for k in range(nsteps + 1):
                c2 = cmath.exp((+1j / hbar) * (t[k] * (w0 + V0 * ng) + 2 * V0 * phi_n[m, k]))
                c3 = (cmath.exp(-1j*V0*t[k])-1)*ng - 1
                c = c2*c3
                s1[m, k] = -c1*c.imag

    s1_mean = np.mean(s1, 0)
    s1dummy = np.ones(nsteps + 1, dtype=complex)  # spectrum without broadening
    s1dummy2 = np.ones(nsteps + 1, dtype=complex)  # spectrum without broadening and without shift
    for k in range(nsteps + 1):
        c2 = cmath.exp((+1j / hbar) * (t[k] * (w0 + V0 * ng) + 2 * V0 * 0))
        c3 = (cmath.exp(-1j*V0*t[k])-1)*ng - 1
        c = c2*c3
        s1dummy[k] = -c1*c.imag
        s1dummy2[k] = c1*(cmath.exp(+1j*t[k]*w0/hbar)).imag

    plt.figure()
    plt.plot(t, s1_mean)
    #plt.plot(t, s1dummy)
    #plt.plot(t, s1dummy2)
    #plt.show()

    spec_s1 = fftpack.fftshift(fftpack.fft(s1_mean))
    spec_s1dummy = fftpack.fftshift(fftpack.fft(s1dummy))
    spec_s1dummy2 = fftpack.fftshift(fftpack.fft(s1dummy2))
    freq = fftpack.fftshift(fftpack.fftfreq(t.shape[-1], dt)) * (2 * np.pi)*hbar

    plt.figure()
    plt.plot(freq, abs(spec_s1dummy2))
    plt.plot(freq, abs(spec_s1dummy))
    plt.plot(freq, abs(spec_s1))
    plt.xlabel('Energy (eV)')
    plt.legend(['Original', 'Interacting', 'Stochastic'])
    plt.show()

