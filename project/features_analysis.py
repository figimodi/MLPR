from mllib import *

if __name__ == '__main__':
    (DTR, LTR) = load('Train.txt')

    # Showing for each feature the distribution of this with respect to the two different classes
    for i in range(DTR.shape[0]):
        feature_plot_binary(i, DTR, LTR, ['spoofed', 'authentic'])

    PCA_plot(DTR, LTR)
    LDA_plot(DTR, LTR)

    heatmaps_binary(DTR, LTR)
