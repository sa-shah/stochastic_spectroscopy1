import time
import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
    start = time.perf_counter()
    import finite_difference
    #finite_difference.first_order_spec()
    for x in [0.25, 0.125, 0.04, 0.0025]:
        e, S1 = finite_difference.first_order_exact(sigmaN0=x**0.5, N0=2)
        #e2, S2 = finite_difference.first_order_spec(N0=2)
        e3, S3 = finite_difference.laserbroadening(sigmaN0=x**0.5, N0=2, samples=1000)
        np.save('Results/laser_broadening_energy_sigmaN0_'+str(x), e3)
        np.save('Results/laser_broadening_spec_sigmaN0_', S3)
        np.save('Results/first_order_exact_energy_sigmaN0_'+str(x), e)
        np.save('Results/first_order_exact_spec_sigmaN0_'+str(x), S1)
        np.save('Results/all_results_sigmaN0_'+str(x), np.array([e, S1, e3, S3]))

    for x in [0, 2, 4, 6]:
        e, S1 = finite_difference.first_order_exact(N0=x)
        #e2, S2 = finite_difference.first_order_spec(N0=2)
        e3, S3 = finite_difference.laserbroadening(N0=x, samples=1000)
        np.save('Results/laser_broadening_energy_N0_'+str(x), e3)
        np.save('Results/laser_broadening_spec_N0_', S3)
        np.save('Results/first_order_exact_energy_N0_'+str(x), e)
        np.save('Results/first_order_exact_spec_N0_'+str(x), S1)
        np.save('Results/all_results_N0_'+str(x), np.array([e, S1, e3, S3]))

    for x in [50, 15, 10, 5, 2, 1, 0.1]:
        e, S1 = finite_difference.first_order_exact(N0=4, gamma=x/1000)
        # e2, S2 = finite_difference.first_order_spec(N0=2)
        e3, S3 = finite_difference.laserbroadening(N0=4, gamma=x/1000, samples=1000)
        np.save('Results/laser_broadening_energy_N0_2_gamma_' + str(x), e3)
        np.save('Results/laser_broadening_spec_N0_2_gamma_', S3)
        np.save('Results/first_order_exact_energy_N0_2_gamma_' + str(x), e)
        np.save('Results/first_order_exact_spec_N0_2_gamma_' + str(x), S1)
        np.save('Results/all_results_N0_2_gamma_' + str(x), np.array([e, S1, e3, S3]))

    finish = time.perf_counter()
    print(f'time taken {finish-start} seconds')
    # plt.figure()
    # plt.plot(e, S1)
    # #plt.plot(e2, S2)
    # plt.plot(e3, S3)
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('S1(w)')
    # plt.title('A comparison of analytical and numerical solutions: N0=2, dt=0.1, gamma=0.012 eV sigma=0.0025**0.5, sigmaN0=0.125**0.5')
    # plt.legend(['analytical', 'Numerical with 1 run (1000 trials)', 'Numerical with 100 runs (1000 trials)'])
    # plt.show()
    # print(f"time taken {finish-start}")
    # print('office pc branch')


