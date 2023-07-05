from mllib import *

if __name__ == '__main__':
    D, L = load('Train.txt')
    
    # DPCA9 = PCA(D, L, 9)
    # DPCA8 = PCA(D, L, 8)
    DPCA7 = PCA(D, L, 7)
    # DPCA6 = PCA(D, L, 6)

    zScoreD = z_score(DPCA7)
    expD = expand_feature_space(zScoreD)

    # folds
    K = 10

    # lambda
    l = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    # threshold
    p = 1/11
    t = -np.log(p/(1-p))

    for li in l:
         # accuracy
        acc = 0
        logRatioCumulative = np.array([])
        cumulativeLabels = np.array([])

        for i in range(0, K):
            (DTR, LTR), (DTE, LTE) = k_fold(expD, L, K, i)

            # use maxfun=[>1500], maxiter[>30], factr=[<10**7] to increment precision
            x0 = np.zeros(DTR.shape[0] + 1)
            x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj_weight_wrap(DTR, LTR, li, 0.5), x0)

            w, b = x[0:-1], x[-1]
            S = np.dot(w, DTE) + b

            empp = LTR.sum()/LTR.shape[0]
            logempp = np.log(empp/(1 - empp))

            PL = S > t

            logRatioCumulative = np.append(logRatioCumulative, S)
            cumulativeLabels = np.append(cumulativeLabels, LTE)

            acc += (PL == LTE).sum()

        acc /= len(L)
        err = 1 - acc

        dcf = normalized_bayes_risk(p, 1, 1, logRatioCumulative, cumulativeLabels)
        mindcf = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)

        print(f"using lambda={li}")
        print(f"accuray: {acc}")
        print(f"min dcf: {mindcf}")
        print(f"actual dcf: {dcf}") 
        print("___________________________________")