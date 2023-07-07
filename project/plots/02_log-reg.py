import matplotlib.pyplot as plt 

if __name__ == '__main__':
    x = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    lin_nopca = [0.4836885245901639, 0.4836885245901639, 0.4836885245901639, 0.4836885245901639, 0.4821311475409836, 0.481188524590164, 0.5393032786885246]
    lin_nopca_z = [0.4836885245901639, 0.4836885245901639, 0.4836885245901639, 0.48243852459016395, 0.5080532786885246, 0.5564549180327869, 0.6883811475409836]
    lin_pca7 = [0.46961065573770494, 0.46961065573770494, 0.46961065573770494, 0.46961065573770494, 0.4746106557377049, 0.4824180327868852, 0.5330327868852459]
    lin_pca7_z = [0.46961065573770494, 0.46961065573770494, 0.46836065573770497, 0.4589959016393443, 0.4674180327868852, 0.485860655737705, 0.7385860655737705]

    quad_nopca = [0.2924180327868852, 0.29366803278688525, 0.29366803278688525, 0.28991803278688527, 0.29303278688524587, 0.31614754098360653, 0.3314139344262295]
    quad_pca9 = [0.2842622950819672, 0.28301229508196724, 0.28680327868852457, 0.2864754098360655, 0.2880327868852459, 0.3214344262295082, 0.33360655737704914]
    quad_pca8 = [0.28397540983606556, 0.2864754098360655, 0.28522540983606554, 0.28489754098360653, 0.2864549180327869, 0.3164344262295082, 0.33799180327868855]
    quad_pca7 = [0.2655532786885246, 0.26680327868852455, 0.2643032786885246, 0.2618032786885246, 0.28428278688524594, 0.32112704918032786, 0.3311065573770492]
    quad_pca6 = [0.2689959016393443, 0.2689959016393443, 0.2689959016393443, 0.26961065573770493, 0.2889549180327869, 0.3104918032786885, 0.32735655737704916]

    quad_nopca_z = [0.29272540983606554, 0.2924180327868852, 0.2886885245901639, 0.2805327868852459, 0.28897540983606557, 0.32739754098360657, 0.3905122950819672]
    quad_pca7_z = [0.2643032786885246, 0.2643032786885246, 0.2593032786885246, 0.2636680327868853, 0.30645491803278685, 0.3402254098360656, 0.4333196721311475]
    quad_pca7_zw = [0.26680327868852455, 0.26680327868852455, 0.2643032786885246, 0.2643032786885246, 0.26836065573770496, 0.29489754098360654, 0.4252254098360656]
    quad_pca7_zwl = [0.2708606557377049, 0.2586680327868852, 0.25366803278688527, 0.31741803278688524, 0.4158811475409836, 0.5014344262295082, 0.921844262295082]
    quad_pca7_cwl = [0.2695901639344262, 0.26147540983606554, 0.2768032786885246, 0.34239754098360653, 0.45932377049180323, 0.5286065573770492, 0.9109016393442623]

    quad_pca7_z_weight = [0.2580532786885246, 0.2555532786885246, 0.2618032786885246, 0.2611680327868853, 0.3052049180327869, 0.3515163934426229, 0.4034016393442623]
    quad_pca7_zwl_weight = [0.2639754098360656, 0.2593032786885246, 0.2571106557377049, 0.3161680327868852, 0.4305532786885246, 0.40241803278688526, 0.44643442622950813]

    plt.figure()
    plt.plot(x, lin_nopca, label='Log-Reg')
    plt.plot(x, lin_nopca_z, label='Log-Reg (z-norm)')
    plt.plot(x, lin_pca7, label='Log-Reg (PCA=7)', linestyle='--', linewidth=2)
    plt.plot(x, lin_pca7_z, label='Log-Reg (PCA=7, z-norm)', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('log-reg\\linear_logreg.png')
    plt.close

    plt.figure()
    plt.plot(x, quad_nopca, label='Q-Log-Reg')
    plt.plot(x, quad_nopca_z, label='Q-Log-Reg (z-norm)')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('log-reg\\quad.png')
    plt.close()

    plt.figure()
    plt.plot(x, quad_nopca, label='Q-Log-Reg')
    plt.plot(x, quad_pca9, label='Q-Log-Reg (PCA=9)')
    plt.plot(x, quad_pca8, label='Q-Log-Reg (PCA=8)')
    plt.plot(x, quad_pca7, label='Q-Log-Reg (PCA=7)')
    plt.plot(x, quad_pca6, label='Q-Log-Reg (PCA=6)')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('log-reg\\quad_diffPCA.png')
    plt.close()

    plt.figure()
    plt.plot(x, quad_pca7, label='Q-Log-Reg', linewidth=3)
    plt.plot(x, quad_pca7_z, label='Q-Log-Reg (z-norm)')
    plt.plot(x, quad_pca7_zw, label='Q-Log-Reg (z-norm, whitening)')
    plt.plot(x, quad_pca7_zwl, label='Q-Log-Reg (z-norm, whitening, l2-norm)')
    plt.plot(x, quad_pca7_cwl, label='Q-Log-Reg (centering, whitnening, l2-norm)')
    plt.ylim(0.2, 0.6)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('log-reg\\quad_preproc.png')
    plt.close()

    plt.figure()
    plt.plot(x, quad_pca7_z, label='Q-Log-Reg (z-norm)')
    plt.plot(x, quad_pca7_zwl, label='Q-Log-Reg (z-norm, whitening, l2-norm)')
    plt.plot(x, quad_pca7_z_weight, label='Q-Log-Reg (z-norm) prior weighted', linestyle='--')
    plt.plot(x, quad_pca7_zwl_weight, label='Q-Log-Reg (z-norm, whitnening, l2-norm) prior weighted', linestyle='--')
    plt.ylim(0.2, 0.6)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('log-reg\\quad_priorweight.png')
    plt.close()

