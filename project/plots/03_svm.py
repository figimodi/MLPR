import matplotlib.pyplot as plt 

if __name__ == '__main__':
    x = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    lin_nopca = [1.0, 0.9239959016393442, 0.7420286885245901, 0.5104508196721311, 0.47307377049180327, 0.4721311475409837, 0.4946311475409836, 0.97125]
    lin_nopca_z = [1.0, 0.5661270491803279, 0.4980327868852459, 0.47305327868852454, 0.4721311475409837, 0.4721311475409837, 0.4746106557377049, 0.46838114754098364]
    lin_nopca_zwl = [1.0, 0.5273565573770491, 0.5039139344262296, 0.3839754098360656, 0.3505122950819672, 0.3442622950819672, 0.3367622950819672, 0.3367622950819672]
    lin_pca7 = [1.0, 0.929344262295082, 0.7676024590163935, 0.5204918032786885, 0.4758811475409836, 0.4674385245901639, 0.4948975409836066, 0.9865573770491805]
    lin_pca7_z = [1.0, 0.4686680327868853, 0.4771311475409836, 0.4755532786885246, 0.4649385245901639, 0.4674385245901639, 0.47305327868852454, 0.473688524590164]
    lin_pca7_zwl = [1.0, 0.5699180327868852, 0.5392827868852459, 0.36432377049180326, 0.3377254098360655, 0.34178278688524594, 0.34147540983606556, 0.3392827868852459]

    kern_nopca_p2_c0 = [1.0, 0.31739754098360656, 0.3020901639344262, 0.2980327868852459, 0.2945901639344263, 0.67625, 0.9974999999999999, 0.9974999999999999]
    kern_nopca_p2_c1 = [1.0, 0.3036475409836066, 0.3020901639344262, 0.2983401639344262, 0.2989754098360655, 0.4579713114754098, 0.99875, 1.0]
    kern_nopca_p2_c0_zwl = [1.0, 0.46961065573770494, 0.42454918032786876, 0.3367622950819672, 0.28116803278688524, 0.2714754098360656, 0.2730532786885246, 0.3045491803278688]
    kern_nopca_p2_c1_zwl = [1.0, 0.4918032786885246, 0.3855122950819672, 0.31831967213114754, 0.28116803278688524, 0.2680532786885246, 0.2718032786885246, 0.3045491803278688]
    
    kern_pca7_p2_c0 = [1.0, 0.30676229508196723, 0.2989754098360655, 0.30086065573770493, 0.2846106557377049, 0.688217213114754, 0.99875, 1.0]
    kern_pca7_p2_c1 = [1.0, 0.2996106557377049, 0.2961475409836065, 0.30084016393442625, 0.29770491803278687, 0.49575819672131144, 0.9831147540983606, 1.0]
    kern_pca7_p2_c0_zwl = [1.0, 0.46241803278688526, 0.4223975409836066, 0.3064754098360656, 0.26586065573770495, 0.2518032786885246, 0.27524590163934426, 0.2733606557377049]
    kern_pca7_p2_c1_zwl = [1.0, 0.5111680327868853, 0.39274590163934425, 0.3014754098360655, 0.2671106557377049, 0.25305327868852456, 0.2643032786885246, 0.26961065573770493]

    kern_nopca_p3_c0 = [1.0, 0.590983606557377, 0.7116188524590165, 0.686577868852459, 0.6887295081967214, 0.6803073770491803, 0.6803073770491803, 0.6803073770491803]
    kern_nopca_p3_c1 = [1.0, 0.5991598360655738, 0.6991598360655737, 0.6897336065573771, 0.6839959016393443, 0.6806147540983606, 0.6806147540983606, 0.6806147540983606]
    kern_nopca_p3_c0_zwl = [1.0, 0.4304918032786885, 0.36053278688524587, 0.2924180327868852, 0.26991803278688525, 0.2711680327868852, 0.2945901639344263, 0.38170081967213115]
    kern_nopca_p3_c1_zwl = [1.0, 0.4407991803278688, 0.33618852459016396, 0.2721106557377049, 0.26491803278688525, 0.2702254098360656, 0.3130327868852459, 0.38545081967213113]

    kern_pca7_p3_c0 = [1.0, 0.39547131147540987, 0.3907991803278688, 0.8031967213114753, 1.0, 0.9940573770491805, 1.0, 0.99875]
    kern_pca7_p3_c1 = [1.0, 0.3892418032786885, 0.44170081967213115, 0.8470696721311475, 1.0, 0.98875, 1.0, 1.0]
    kern_pca7_p3_c0_zwl = [1.0, 0.4111680327868853, 0.3505327868852459, 0.2805327868852459, 0.25305327868852456, 0.25618852459016395, 0.3155327868852459, 0.3701844262295082]
    kern_pca7_p3_c1_zwl = [1.0, 0.4533401639344262, 0.31645491803278686, 0.2748975409836066, 0.2543032786885246, 0.25399590163934427, 0.32520491803278684, 0.3686680327868852]

    kern_nopca_p2_c1_k10_zwl = [1.0, 0.3011475409836066, 0.3018032786885246, 0.30743852459016396, 0.3308606557377049, 0.4739549180327869, 1.0, 1.0]
    kern_nopca_p2_c1_k100_zwl = [1.0, 0.2946106557377049, 0.9206147540983607, 0.99875, 0.9915573770491805, 0.9828073770491803, 0.9974999999999999, 1.0]
    kern_pca7_p2_c1_k10_zwl = [1.0, 0.35866803278688525, 0.33584016393442623, 0.31709016393442624, 0.26961065573770493, 0.2714959016393443, 0.8173975409836067, 0.9737295081967213]
    kern_pca7_p2_c1_k100_zwl = [1.0, 0.597438524590164, 0.8993647540983606, 0.965, 0.9706147540983606, 0.9690573770491805, 1.0, 0.9368647540983606]

    kern_nopca_p3_c1_k10_zwl = [1.0, 0.676311475409836, 0.670327868852459, 0.670327868852459, 0.670327868852459, 0.670327868852459, 0.670327868852459, 0.670327868852459]
    kern_nopca_p3_c1_k100_zwl = [1.0, 0.9962499999999999, 0.9974999999999999, 0.9837295081967214, 0.9962499999999999, 1.0, 0.995, 0.9587295081967213]
    kern_pca7_p3_c1_k10_zwl = [1.0, 0.27866803278688523, 0.2568032786885246, 0.5642008196721312, 0.9256147540983607, 0.99875, 0.99, 1.0]
    kern_pca7_p3_c1_k100_zwl = [1.0, 0.9340573770491803, 0.9925000000000002, 0.9475, 0.9974999999999999, 0.9406147540983606, 0.9974999999999999, 0.9709221311475411]

    kern_pca7_g1_zwl = []
    kern_pca7_g10_zwl = []
    kern_pca7_g100_zwl = []

    plt.figure()
    plt.plot(x, lin_nopca, label='SVM (No PCA)')
    plt.plot(x, lin_nopca_z, label='SVM (No PCA, z-norm)')
    plt.plot(x, lin_nopca_zwl, label='SVM (No PCA, z-norm + whitening + l2-norm)')
    plt.plot(x, lin_pca7, label='SVM (PCA = 7)', linestyle='--', linewidth=2)
    plt.plot(x, lin_pca7_z, label='SVM (PCA = 7, z-norm)', linestyle='--', linewidth=2)
    plt.plot(x, lin_pca7_zwl, label='SVM (PCA = 7, z-norm + whitening + l2-norm)', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('svm\\linear_svm.png')
    plt.close

    plt.figure()
    plt.plot(x, kern_nopca_p2_c0, label='SVM - poly(2) (No PCA, c = 0)')
    plt.plot(x, kern_nopca_p2_c1, label='SVM - poly(2) (No PCA, c = 1)')
    plt.plot(x, kern_nopca_p2_c0_zwl, label='SVM - poly(2) (No PCA, z-norm + whitening + l2-norm, c = 0)')
    plt.plot(x, kern_nopca_p2_c1_zwl, label='SVM - poly(2) (No PCA, z-norm + whitening + l2-norm, c = 1)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('svm\\kern_svm_poly2_nopca.png')
    plt.close

    plt.figure()
    plt.plot(x, kern_pca7_p2_c0, label='SVM - poly(2) (PCA=7, c = 0)')
    plt.plot(x, kern_pca7_p2_c1, label='SVM - poly(2) (PCA=7, c = 1)')
    plt.plot(x, kern_pca7_p2_c0_zwl, label='SVM - poly(2) (PCA=7, z-norm + whitening + l2-norm, c = 0)')
    plt.plot(x, kern_pca7_p2_c1_zwl, label='SVM - poly(2) (PCA=7, z-norm + whitening + l2-norm, c = 1)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('svm\\kern_svm_poly2_pca7.png')
    plt.close

    plt.figure()
    plt.plot(x, kern_nopca_p3_c0, label='SVM - poly(3) (No PCA, c = 0)')
    plt.plot(x, kern_nopca_p3_c1, label='SVM - poly(3) (No PCA, c = 1)')
    plt.plot(x, kern_nopca_p3_c0_zwl, label='SVM - poly(3) (No PCA, z-norm + whitening + l2-norm, c = 0)')
    plt.plot(x, kern_nopca_p3_c1_zwl, label='SVM - poly(3) (No PCA, z-norm + whitening + l2-norm, c = 1)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('svm\\kern_svm_poly3_nopca.png')
    plt.close

    plt.figure()
    plt.plot(x, kern_pca7_p3_c0, label='SVM - poly(3) (PCA=7, c = 0)')
    plt.plot(x, kern_pca7_p3_c1, label='SVM - poly(3) (PCA=7, c = 1)')
    plt.plot(x, kern_pca7_p3_c0_zwl, label='SVM - poly(3) (PCA=7, z-norm + whitening + l2-norm, c = 0)')
    plt.plot(x, kern_pca7_p3_c1_zwl, label='SVM - poly(3) (PCA=7, z-norm + whitening + l2-norm, c = 1)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('svm\\kern_svm_poly3_pca7.png')
    plt.close

    plt.figure()
    plt.plot(x, kern_nopca_p2_c1_k10_zwl, label='SVM - poly(2) (NO PCA, K=10)')
    plt.plot(x, kern_nopca_p2_c1_k100_zwl, label='SVM - poly(2) (NO PCA, K=100)')
    plt.plot(x, kern_pca7_p2_c1_k10_zwl, label='SVM - poly(2) (PCA=7, K=10)')
    plt.plot(x, kern_pca7_p2_c1_k10_zwl, label='SVM - poly(2) (PCA=7, K=100)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('svm\\kern_svm_poly2_Ks.png')
    plt.close

    plt.figure()
    plt.plot(x, kern_nopca_p3_c1_k10_zwl, label='SVM - poly(3) (NO PCA, K=10)')
    plt.plot(x, kern_nopca_p3_c1_k100_zwl, label='SVM - poly(3) (NO PCA, K=100)')
    plt.plot(x, kern_pca7_p3_c1_k10_zwl, label='SVM - poly(3) (PCA=7, K=10)')
    plt.plot(x, kern_pca7_p3_c1_k10_zwl, label='SVM - poly(3) (PCA=7, K=100)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('svm\\kern_svm_poly3_Ks.png')
    plt.close
