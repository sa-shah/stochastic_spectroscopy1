def mc_container():
    n_sum = dt*np.cumsum(n, axis=-1) #effectively this is the integral of each trajectory (through finite element menthod)
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


def first_order_spec():
    trials = 1000 # number of simulations to run for each starting value of N0
    samples = 100 #number samples from the distribution of N0
    dt = 0.001
    nsteps = 10000
    #gamma = 0.01 # per fs
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