import time
import matplotlib.pyplot as plt
if __name__=='__main__':
    start = time.perf_counter()
    import finite_difference
    #finite_difference.first_order_spec()
    e, S1 = finite_difference.first_order_exact(sigmaN0=0, N0=2)
    e2, S4 = finite_difference.first_order_spec(N0=2)
    finish = time.perf_counter()
    plt.figure()
    plt.plot(e, S1)
    plt.plot(e,S4)
    plt.xlabel('Energy (eV)')
    plt.ylabel('S1(w)')
    plt.title('A comparison of analytical and numerical solutions: N0=2, dt=0.1, gamma=0.012 eV sigma=0.0025**0.5, sigmaN0=0')
    plt.legend(['analytical','numerical (1000-trial avg.)'])
    plt.show()
    print(f"time taken {finish-start}")
    print('office pc branch')


