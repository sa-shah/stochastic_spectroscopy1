
from numpy import random, random_intel
import matplotlib.pyplot as plt
import numpy as np

def finite_difference_method(n0 = None, nsteps=1000, dt=0.01, trials = 100, gamma = 0.15, sigma = (0.0025)**0.5):
    """ this function computes the numerical intergral of a stochastic process
    through the first order approximation from taylor expansion
    u((i+1)h) = u(ih) + h*u'(ih)
    """

    if n0 is None:
        n0 = np.ones(trials)*10

    # Create an empty array to store the realizations.
    x = np.empty((trials, nsteps + 1))
    # Initial values of x.
    x[:, 0] = 0 #
    xx,dB = brownian(x[:, 0], nsteps, dt, out=x[:, 1:])
    print(dB.shape, 'is the shape of db')
    print('shape of x is ', x.shape, ' 1st element ', x[0][0])
    t = np.linspace(0.0, nsteps * dt, nsteps + 1)
    print('the shape of t is ', t.shape)
    n = np.ones((trials, nsteps+1))*1
    print('shape of n is ',n.shape)
    for m in range(trials):
        for h in range(nsteps):
            n[m,h+1] = n[m,h]+1*(-gamma*n[m,h]*dt+sigma*dB[m,h])
    print('new shape of n is ', n.shape)
    print(n)
    plt.figure()
    for k in range(trials):
        plt.plot(t, n[k])
    plt.plot(t,np.mean(n,0),linewidth=6)
    plt.show()

    plt.figure()
    for k in range(trials):
        plt.plot(t, x[k])
    plt.plot(t, [(k ** 0.5) for k in t], linewidth=6)
    plt.plot(t, [-(k ** 0.5) for k in t], linewidth=6)
    plt.plot(t, t, linewidth=6)
    plt.plot(t, -t, linewidth=6)
    plt.show()

    return None




def brownian(x0, n, dt, out=None):
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

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    #r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))
    r = random.normal(0,dt**0.5,x0.shape+(n,))
    #size is the number of random values, scale is the std of distribution, and loc = location (mu)
    #plt.figure()
    #plt.hist(np.squeeze(r))
    #plt.show()
    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out,r

finite_difference_method()


#print(dB.shape, 'shape of db', dB[0][:10])
#plt.figure()
#plt.hist(dB[0][:])
#plt.show()