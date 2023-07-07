import scipy as sp
import numpy as np
import sklearn.datasets


def svm_wraper(H):
    def svm_obj(alpha, DTR):
        # we now need to return the objective L = -J

        LD = 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.dot(alpha.T, np.ones((DTR.shape[1], 1)))
        LDG = np.reshape(np.dot(H, alpha) - np.ones((1, DTR.shape[1])), (DTR.shape[1],1))
        
        return (LD, LDG)
    
    return svm_obj

def compute_svm(DTR, LTR, DTE, K, C):

    Z = LTR * 2 - 1
    DTRE = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    D = np.multiply(DTRE, Z.T)
    H = np.dot(D.T, D)

    # define the array of constraints for the objective
    BC = [(0, C) for i in range(0, DTR.shape[1])]
    [alpha, LD, d] = sp.optimize.fmin_l_bfgs_b(svm_wraper(H), np.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
    
    
    # need to compute the primal solution from the dual solution
    w = np.multiply(alpha, np.multiply(DTRE, Z.T)).sum(axis=1)

    # need to compute the duality gap
    S = -np.dot(w.T, D) + 1
    JP = 0.5 * (np.linalg.norm(w) ** 2) + C * (S[S>0]).sum()
    
    print("The duality gap is ", JP + LD)

    # we now need to compute the scores and check the predicted lables with threshold
    DTEE = np.vstack([DTE, np.ones((1, DTE.shape[1])) * K])
    return np.dot(w.T, DTEE)

def poly_kernel(x1, x2, c, d, e):
    return np.power((np.dot(x1.T, x2) + c), d) + e
    
def compute_svm_polykernel(DTR, LTR, DTE, K, C, d, c):
    Z = LTR * 2 - 1
    DTRE = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    
    Z = np.reshape(Z, (LTR.shape[0], 1))
    H = np.dot(Z, Z.T)

    # will compute H in with for loops
    for i in range(0, DTR.shape[1]):
        for j in range(0, DTR.shape[1]):
            H[i][j] *= poly_kernel(DTRE.T[i], DTRE.T[j], c, d, K**2)

    BC = [(0, C) for i in range(0, DTR.shape[1])]
    [alpha, f, d2] = sp.optimize.fmin_l_bfgs_b(svm_wraper(H), np.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
    
    DTEE = np.vstack([DTE, np.ones((1, DTE.shape[1])) * K])

    S = np.ones((DTE.shape[1]))

    for i in range(0, DTE.shape[1]):
        result = 0
        for j in range(0, DTR.shape[1]):
            result += alpha[j]*Z[j]*poly_kernel(DTRE.T[j], DTEE.T[i], c, d, K**2)
        S[i] = result
    
    return S


if __name__ == '__main__':
    # [D, L] = load()
    # [DTR, LTR, DTE, LTE] = split_db_2to1(D, L)
    
    # svm_wraper(DTR, LTR, 1)(np.zeros((DTR.shape[1],1)))

    # L = compute_svm(DTR, LTR, 1, 0.1, DTE)
    # print("The error rate is --> ", 100*(1 - (L == LTE).sum()/(LTE.shape[0])))

    # LK = compute_svm_polykernel(DTR, LTR, 0, 1, DTE, 2, 1)
    # print("The error rate is --> ", 100*(1 - (LK == LTE).sum()/(LTE.shape[0])))
    print("Hello")