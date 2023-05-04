import numpy as np
import scipy as sp

def vcol(mat):
    return mat.reshape((mat.size, 1)) 

def vrow(mat):
    return mat.reshape((1, mat.size))

def load(file_name):
    Dlist = []
    L = []

    with open(file_name) as f:
        for line in f:
            x = vcol(np.array([float(i) for i in line.split(',')[0:10]]))
            Dlist.append(x)
            l = int(line.split(',')[-1][0])
            L = np.hstack([L, l])

    return np.hstack(Dlist), L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def compute_mu_C(D, L, label, NB=False):
    DL = D[:, L == label]
    mu = DL.mean(1).reshape(DL.shape[0], 1)
    DLC = (DL - mu)
    C = 1/DLC.shape[1]*np.dot(DLC, DLC.T)

    if NB:
        C = np.multiply(C, np.identity(DL.shape[0]))

    return (mu, C)   

def logpdf_GAU_ND(X, mu, C):
    # X array of shape(M, N)
    # mu array of shape (M, 1)
    # C array of shape (M, M) that represents the covariance matrix
    M = C.shape[0] #number of features
    # N = X.shape[1] #number of samples
    invC = np.linalg.inv(C) #C^-1
    logDetC = np.linalg.slogdet(C)[1] #log|C|
    
    # with the for loop:
    # logN = np.zeros(N)
    # for i, sample in enumerate(X.T):
    #     const = -0.5*M*np.log(2*np.pi)
    #     dot1 = np.dot((sample.reshape(M, 1) - mu).T, invC)
    #     dot2 = np.dot(dot1, sample.reshape(M, 1) - mu)
    #     logN[i] = const - 0.5*logDetC - 0.5*dot2

    XC = (X - mu).T # XC has shape (N, M)
    const = -0.5*M*np.log(2*np.pi)

    # sum(1) sum elements of the same row togheter
    # multiply make an element wise multiplication
    logN = const - 0.5*logDetC - 0.5*np.multiply(np.dot(XC, invC), XC).sum(1)

    # logN is an array of length N (# of samples)
    # each element represents the log-density of each sample
    return logN

def PCA(D, L, m=2):
    mu = D.mean(1) # mu will be a row vector so we have to convert it into a column vector
    Dc = D - vcol(mu) # centered dataset D - the column representation of mu
    C = (1/D.shape[1])*np.dot(Dc, Dc.T) # C is the covariance matrix

    # find eigenvalues and eigenvectors with the function numpy.linalg.eigh
    s, U = np.linalg.eigh(C) # eigh returns the eigenvalues and the eignevectors sorted from smallest to larger

    # find eigenvalues and eigenvectors with SVD 
    #U, s, Vh = np.linalg.svd(C)

    # [:, ::-1] takes all the columns and sort them in reverse order
    # [:, 0:m] takes all the row from 0 to index m
    P = U[:, ::-1][:, 0:m]
    Dp = np.dot(P.T, D) # project the dataset to the new space
    
    # create the matrix related to the specific classes
    D0 = Dp[:, L==0]
    D1 = Dp[:, L==1]

    # Create the figure
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'PCA for {m} components')

    if m == 2:
        plt.scatter(D0[0], D0[1], label='Spoofed')
        plt.scatter(D1[0], D1[1], label='Authentic')

    if m == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(D0[0], D0[1], D0[2], label='Spoofed')
        ax.scatter(D1[0], D1[1], D1[2], label='Authentic')

    return Dp

def LDA(D, L, m=2):
    # create the matrix related to the specific classes
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    # compute the average of each class and the total one
    mu = vcol(D.mean(1))
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))

    # compute centered datasets
    Dc0 = D0 - mu0
    Dc1 = D1 - mu1

    # compute the covariance matrices
    C0 = (1/D0.shape[1])*np.dot(Dc0, Dc0.T)
    C1 = (1/D1.shape[1])*np.dot(Dc1, Dc1.T)
    
    # within class covariance matrix
    Sw = (1/D.shape[1])*(D0.shape[1]*C0 + D1.shape[1]*C1)

    # between class covariance matrix
    Sb0 = D0.shape[1]*np.dot(mu0 - mu, (mu0 - mu).T)
    Sb1 = D1.shape[1]*np.dot(mu1 - mu, (mu1 - mu).T)
    Sb = (1/D.shape[1])*(Sb0 + Sb1)

    # solve the generalized eigenvalue problem Sb*w=lambda*Sw*w with sp.linalg.eigh
    s, U = sp.linalg.eigh(Sb, Sw)
    
    # get W as the first m eigenvectors of U
    W = U[:, ::-1][:, 0:m]

    # since the columns of W are not necessary orthogonal we can find a base U using SVD
    #Uw, s, Vh = np.linalg.svd(W)
    #U = Uw[:, 0:m]

    Dp = np.dot(W.T, D) # project the dataset to the new space
    
    # create the matrix related to the specific classes
    D0 = Dp[:, L==0]
    D1 = Dp[:, L==1]

    # Create the figure
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'LDA for {m} components')

    if m == 2:
        plt.scatter(D0[0], D0[1], label='Spoofed')
        plt.scatter(D1[0], D1[1], label='Authentic')

    if m == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(D0[0], D0[1], D0[2], label='Spoofed')
        ax.scatter(D1[0], D1[1], D1[2], label='Authentic')

    return Dp

def k_fold(D, L, K, i, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTest = idx[int(i*D.shape[1]/K):int((i+1)*D.shape[1]/K)]
    idxTrain0 = idx[:int(i*D.shape[1]/K)]
    idxTrain1 = idx[int((i+1)*D.shape[1]/K):]
    idxTrain = np.hstack([idxTrain0, idxTrain1])
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)