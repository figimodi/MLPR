import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    k = int(10e5)

    x = [1, 3, 5, 2]
    y = [1, 5, 1, 2]

    for i in range(0, k):
        z = np.random.randint(0, 3)
        x_a, y_a = x[z], y[z]
        x_b, y_b = x[-1], y[-1]
        x_n, y_n = x_b + (x_a-x_b)/2, y_b + (y_a-y_b)/2
        x.append(x_n)
        y.append(y_n)

    plt.scatter(x, y, marker='.', s=1)    
    plt.show()