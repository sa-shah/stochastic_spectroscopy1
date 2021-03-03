import time
import matplotlib.pyplot as plt
if __name__=='__main__':
    start = time.perf_counter()
    import finite_difference
    finite_difference.first_order_spec()
    # t, S1 = finite_difference.first_order_exact(sigmaN0=0.25**0.5, N0=0)
    # t, S2 = finite_difference.first_order_exact(sigmaN0=0.125**0.5, N0=0)
    # t, S3 = finite_difference.first_order_exact(sigmaN0=0*0.04 ** 0.5, N0=2)
    # finish = time.perf_counter()
    # plt.figure()
    # plt.plot(t, S1)
    # plt.plot(t, S2)
    # plt.plot(t, S3)
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('S1(w)')
    # plt.title('N0=0 A comparison to A-K model')
    # plt.legend(['0.25', '0.125', '0.04'])
    # plt.show()
    print(f"time taken {finish-start}")
    print('office pc branch')


