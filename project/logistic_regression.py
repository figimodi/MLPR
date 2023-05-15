from mllibrary import *

if __name__ == '__main__':
    D, L = load('Train.txt')

    # folds
    K = 5

    # lambda
    l = 1e-3

    # threshold
    t = 0

    # accuracy
    acc = 0

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(D, L, K, i)

        logreg_obj = logreg_obj_wrap(DTR, LTR, l)

        # use maxfun=[>1500], maxiter[>30], factr=[<10**7] to increment precision
        x0 = np.zeros(DTR.shape[0] + 1)
        x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, factr=10**7, maxfun=15000, maxiter=30)

        w, b = x[0:-1], x[-1]
        S = np.dot(w, DTE) + b
        
        PL = S > t

        acc += (PL == LTE).sum()

    acc /= len(L)
    err = 1 - acc

    print(acc)
    