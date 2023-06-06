from mllibrary import *

if __name__ == '__main__':
    (DTR, LTR) = load('Train.txt')
    (DTE, LTE) = load('Test.txt')

    (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
    (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

    S0 = logpdf_GAU_ND(DTE, mu0, C0)
    S1 = logpdf_GAU_ND(DTE, mu1, C1)

    S = S1 - S0

    # nbr = normalized_bayes_risk(0.5, 1, 10, S, LTE)
    # print(nbr)

    # mindcf = DCF_min(0.5, 1, 1, S, LTE)
    # print(mindcf)

    bayer_error_plots(0.5, 1, 10, S, LTE)
