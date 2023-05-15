from mllibrary import *

if __name__ == '__main__':
    D, L = load('Train.txt')

    # folds
    K = 10

    # accuracy
    acc = 0

    # DPCA2 = PCA(D, L, 2) # call PCA with m=2
    # DPCA3 = PCA(D, L, 3) # call PCA with m=3
    # DLDA2 = LDA(D, L, 2) # call LDA with m=2
    # DLDA3 = LDA(D, L, 3) # call LDA with m=3

    # (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L) # without PCA or/and LDA
    # (DTR, LTR), (DTE, LTE) = split_db_2to1(DPCA2, L) # without PCA or/and LDA
    # (DTR, LTR), (DTE, LTE) = split_db_2to1(DPCA3, L) # without PCA or/and LDA
    # (DTR, LTR), (DTE, LTE) = split_db_2to1(DLDA2, L) # without PCA or/and LDA
    # (DTR, LTR), (DTE, LTE) = split_db_2to1(DLDA3, L) # without PCA or/and LDA 
    
    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(D, L, K, i)

        # MVG
        # compute mean and covariance for all classes
        # (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
        # (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

        # Naive-Bayes
        # compute mean and covariance for all classes
        # (mu0, C0) = compute_mu_C(DTR, LTR, 0, True)
        # (mu1, C1) = compute_mu_C(DTR, LTR, 1, True)

        # Tied-Covariance
        # C0 = C1 = 1/DTR.shape[1]*(C0*(LTR == 0).sum() + C1*(LTR == 1).sum())

        # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
        S0 = logpdf_GAU_ND(DTE, mu0, C0)
        S1 = logpdf_GAU_ND(DTE, mu1, C1)

        # f_c|x
        S = np.vstack([S0, S1])
        
        # working with exp
        # S = np.exp(S)

        # # f_x|c
        # SJoint = 1/3*S
        # SMarginal = vrow(SJoint.sum(0))
        # SPost = SJoint/SMarginal

        # working with logs
        logSJoint = S + 1/2
        logSMarginal = vrow(sp.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = np.exp(logSPost)

        PL = np.argmax(SPost, 0)

        # TODO Compute ratio instead argmax ?

        acc += (PL == LTE).sum()

    acc /= len(L)
    err = 1 - acc

    print(acc)