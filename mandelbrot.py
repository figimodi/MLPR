import numpy as np
import matplotlib.pyplot as plt

# Set the dimensions and resolution of the plot
width, height = 3200, 3200
xmin, xmax = -2.5, 1.5
ymin, ymax = -2, 2

# Generate the complex plane
x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Generate the Mandelbrot set
c = Z
z = np.zeros_like(Z, dtype=complex)
mandelbrot = np.zeros(Z.shape, dtype=int)

for i in range(50000):
    mask = np.abs(z) < 2
    z[mask] = z[mask] * z[mask] + c[mask]
    mandelbrot += mask

# Create the plot
plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot, cmap='binary', extent=(xmin, xmax, ymin, ymax), origin='lower')
plt.title('Mandelbrot Set (Black and White)')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.xticks([])
plt.yticks([])
plt.show()