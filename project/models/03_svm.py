from mllib import *

if __name__ == '__main__':
    D, L = load('Train.txt')
    
    # DPCA9 = PCA(D, L, 9)
    # DPCA8 = PCA(D, L, 8)
    # DPCA7 = PCA(D, L, 7)
    # DPCA6 = PCA(D, L, 6)

    # Dc = centering(DPCA7)
    # Ds = std_variances(Dc)
    # Dw = whitening(Ds, DPCA7)
    # Dl = l2(Dw)

    # C values
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

    # folds
    K = 10

    # parameters of svm
    KSVM = 1
    degree = 2
    c_poly = 0

    # effective prior
    p = 1/11

    for ci in C:
        logRatioCumulative = np.array([])
        cumulativeLabels = np.array([])

        for i in range(0, K):
            (DTR, LTR), (DTE, LTE) = k_fold(D, L, K, i)

            # S = compute_svm(DTR, LTR, DTE, KSVM, ci)
            S = compute_svm_polykernel(DTR, LTR, DTE, KSVM, ci, degree, c_poly)

            logRatioCumulative = np.append(logRatioCumulative, S)
            cumulativeLabels = np.append(cumulativeLabels, LTE)

        mindcf = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)

        print(f"using C={ci}")
        print(f"min dcf: {mindcf}")
        print("___________________________________")