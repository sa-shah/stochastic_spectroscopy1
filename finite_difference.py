
from numpy import random, random_intel
import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy import fftpack


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

    #print(dW.shape)
    #plt.figure()
    #plt.plot(dW[0, :])
    #plt.show()

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
    N_mean = n_0*np.exp(-0.01*t)
    plt.figure()
    for k in range(trials):
        plt.plot(t, n[k])
    plt.plot(t, np.mean(n, 0), linewidth=6)
    plt.plot(t,N_mean, linewidth=6)
    plt.show()

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


def first_order_spec():
    trials = 100 # number of simulations to run for each starting value of N0
    samples = 100 #number samples from the distribution of N0
    dt = 0.001
    nsteps = 10000
    gamma = 0.01 # per fs
    sigma = 0.025**0.5 # per fs
    sigmaN0 = 0.00125**0.5 # per fs
    N0 = random.normal(2, sigmaN0, samples)
    N = np.ones((samples, nsteps+1))
    for k in range(samples):
        t, N_dummy = finite_difference_method(N0[k], nsteps, dt, trials, gamma, sigma)
        N[k,:] = np.mean(N_dummy, axis=0)

    u = 1
    hbar = 0.6582 #eV.fs
    w0 = 2.35 #eV
    #SN = np.mean(np.cumsum(N, axis=-1), axis=0) #Sum or integral of N(t)
    sN = np.cumsum(N, axis=-1)  # Sum or integral of N(t)
    #print('summed N ', SN.shape)
    V0 = 0.01 #Interaction potential (0.01 eV = 10 meV)
    n0 = 2 #k=0 population
    ss = np.zeros((samples, nsteps+1))
    #ss = np.zeros(nsteps + 1)
    c1 = 2 * (u ** 2) / hbar
    for l in range(samples):
        SN = sN[l]
        for k in range(nsteps+1):
            c2 = cmath.exp(1j*t[k]* (w0 + V0*n0 + 2*V0*SN[k]))
            c3 = (cmath.exp(-1j*V0*t[k]) - 1)*n0 - 1
            ss[l][k] = (c1*c2*c3).imag
            #ss[k] = (c1 * c2 * c3).imag
    s = np.mean(ss, axis=0)  # avg spectrum
    #s = ss
    plt.figure()
    plt.plot(t, s)
    plt.show()

    spec = np.fft.fft(s)
    freq = np.fft.fftfreq(t.shape[-1])
    plt.figure()
    plt.plot(abs(spec))
    #plt.plot(spec.real)
    plt.show()


def MC_tests():
    trials = 1000  # number of simulations to run for each starting value of N0
    dt = 0.1
    nsteps = 10000
    hbar = 0.6582  # eV.fs
    gamma = 0.015  # eV
    gamma = gamma/hbar # per fs
    sigma = 0.0025 ** 0.5  # per fs
    n_0 = 4
    t, n = finite_difference_method(n_0, nsteps, dt, trials, gamma, sigma)
    n_sum = dt*np.cumsum(n, axis=-1) #summed over individual trials but no avg. accross trials
    n_mc = np.mean(n, 0) # average accross trials but not summed over
    n_mc_sum = dt*np.cumsum(n_mc) # summed over avg. path
    n_sum_mc = np.mean(n_sum, 0)

    plt.figure()
    plt.plot(t,n_mc_sum)
    plt.plot(t,t)
    #plt.plot(t,n_sum_mc)
    plt.show()

    V0 = 0.1  # eV
    w0 = 2.35  # eV
    a = np.zeros((trials, nsteps+1), dtype=complex)
    for m in range(trials):
        for k in range(nsteps+1):
            a[m, k] = cmath.exp(-1j*t[k]*(w0 + V0*n_0 + 2*V0*n_sum[m,k]))
    a_mean = np.mean(a, 0)
    A = np.ones(nsteps+1, dtype=complex)
    for k in range(nsteps+1):
        A[k] = cmath.exp(-1j*t[k]*(w0 + V0*n_0 + 2*V0*n_mc_sum[k]))

    adummy = np.ones(nsteps + 1, dtype=complex)
    for k in range(nsteps + 1):
        adummy[k] = cmath.exp(-1j * t[k] * (w0 + V0 * n_0 + 2 * V0 * 1))

    plt.figure()
    plt.plot(t, np.imag(adummy),'r')
    plt.plot(t, np.imag(a_mean),'g')
    plt.plot(t, np.imag(A),'b')
    plt.show()

    #spec_a = np.fft.fft(np.imag(a_mean))
    #spec_A = np.fft.fft(np.imag(A))
    #spec_adummy = np.fft.fft(np.imag(adummy))
    #freq = np.fft.fftfreq(t.shape[-1],dt)/2*np.pi

    spec_a = fftpack.fftshift(fftpack.fft(a_mean))
    spec_A = fftpack.fftshift(fftpack.fft(A))
    spec_adummy = fftpack.fftshift(fftpack.fft(adummy))
    freq = fftpack.fftshift(fftpack.fftfreq(t.shape[-1], dt))*2*np.pi

    plt.figure()
    plt.plot(freq, abs(spec_adummy), 'r')
    plt.plot(freq, abs(spec_a), 'g')
    plt.plot(freq, abs(spec_A), 'b')
    #plt.plot(freq)
    # plt.plot(spec.real)
    plt.show()

#brownian([10, 1.1, 0.9], 10, 0.1)

#print(dB.shape, 'shape of db', dB[0][:10])
#plt.figure()
#plt.hist(dB[0][:])
#plt.show()

#finite_difference_method()
MC_tests()
#first_order_spec()
