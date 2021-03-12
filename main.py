import time
import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
    start = time.perf_counter()
    import finite_difference

    plt.figure()
    for x in [2, 0.25, 0.0025]:
    #checking effect of the variation in initial population.
        e, S1 = finite_difference.first_order_exact(sigmaN0=x**0.5, N0=2)
        #e2, S2 = finite_difference.first_order_spec(N0=2)
        e3, S3 = finite_difference.laserbroadening(sigmaN0=x**0.5, N0=2, samples=100, trials=100)
        np.save('Results/all_results_sigmaN0_' + str(x), np.array([e, S1, e3, S3]))
        plt.plot(e, S1)
        plt.plot(e3, S3)

    plt.title('sigmaN0')
    #plt.legend([str(x) for x in [0.25, 0.125, 0.04, 0.0025]])
    plt.show()
        #np.save('Results/all_results_sigmaN0_'+str(x), np.array([e, S1, e3, S3]))

    # for x in [0, 2, 4, 6]:
    # checking effect of change in the mean value of initial population
    #     e, S1 = finite_difference.first_order_exact(N0=x)
    #     #e2, S2 = finite_difference.first_order_spec(N0=2)
    #     e3, S3 = finite_difference.laserbroadening(N0=x, samples=1000)
    #     np.save('Results/all_results_N0_'+str(x), np.array([e, S1, e3, S3]))
    #
    # for x in [50, 15, 10, 5, 2, 1, 0.1]:
    # checking variation in the decay rate
    #     e, S1 = finite_difference.first_order_exact(N0=4, gamma=x/1000)
    #     # e2, S2 = finite_difference.first_order_spec(N0=2)
    #     e3, S3 = finite_difference.laserbroadening(N0=4, gamma=x/1000, samples=1000)
    #     np.save('Results/all_results_N0_2_gamma_' + str(x), np.array([e, S1, e3, S3]))
    #
    # finish = time.perf_counter()
    # print(f'time taken {finish-start} seconds')

    # plt.figure()
    # for x in [0, 2, 4, 6]:
    #     #plt.figure()
    #     [e1, s1, e3, s3] = np.load('Results/all_results_N0_'+str(x)+'.npy')
    #     plt.plot(e1, s1)
    #     #plt.plot(e3, s3)
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('S1(w)')
    # plt.title('Analytical results for different N0')
    # #plt.legend(['50 A', '50 N', '15 A', '15 N', '10 A', '10 N', '5 A', '5 N', '2 A', '2 N', '1 A', '1 N', '0.1 A', '0.1 N'])
    # #plt.legend(['0 A', '0 N', '2 A', '2 N', '4 A', '4 N', '6 A', '6 N'])
    #     #plt.legend(['Analytical', 'Numerical'])
    # plt.legend([str(x) for x in [0,2,4,6]])
    # plt.show()

