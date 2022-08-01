"""Solving a Laplace problem using spectral methods."""

import numpy as np
import scipy.interpolate as sp
from spectral_shs import cheb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Construct the differentiation matrix
n = 10
D, x = cheb(n)
Dx = np.kron(np.eye(n + 1), D)
Dy = np.kron(D, np.eye(n + 1))

# Construct the grid of points
y = x
xx, yy = np.meshgrid(x, y)
xf = xx.flatten()
yf = yy.flatten()

# Construct the Laplace operator
D2 = D @ D
D2x = np.kron(np.eye(n + 1), D2)
D2y = np.kron(D2, np.eye(n + 1))
L = D2x + D2y
f = np.zeros((n + 1) ** 2)

# Add the boundary conditions
L[np.arange(0, (n + 1) ** 2, n + 1), :] = Dx[
    np.arange(0, (n + 1) ** 2, n + 1), :
]  # Neumann x = 1

L[np.arange(n, (n + 1) ** 2, n + 1), :] = Dx[
    np.arange(n, (n + 1) ** 2, n + 1), :
]  # Neumann x = -1
f[np.arange(n, (n + 1) ** 2, n + 1)] = 1

L[(n + 1) * n :, :] = Dy[(n + 1) * n :, :]  # Neumann y = -1

L[: n + 1, :] = np.eye(n + 1, (n + 1) ** 2)  # Dirichlet y = 1
f[: n + 1] = 1

# Solve the linear system
u = np.linalg.solve(L, f)
uu = u.reshape((n + 1, n + 1))

# Create a finer grid for plotting
x_fine = np.linspace(-1, 1, 41)
y_fine = x_fine
xxx, yyy = np.meshgrid(x_fine, y_fine)
uuu = sp.interp2d(xx, yy, uu, kind="cubic")(x_fine, y_fine)

# Plot the solution
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(xxx, yyy, uuu, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.show()
