"""Solving a Laplace problem on a streched domain of unequal grid sizes."""

import numpy as np
import scipy.interpolate as sp
from spectral_shs import cheb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Construct the differentiation matrix
n = 10
D, x = cheb(n)
Dx = np.kron(np.eye(2 * n + 1), D)
D_double, z = cheb(2 * n)
Dz = np.kron(D_double, np.eye(n + 1))

# Construct the grid of points
xx, zz = np.meshgrid(x, z)
xf = xx.flatten()
zf = zz.flatten()

# Construct the Laplace operator
D2 = D @ D
D2x = np.kron(np.eye(2 * n + 1), D2)
D2_double = D_double @ D_double
D2z = np.kron(D2_double, np.eye(n + 1))
L = D2x + 0.25 * D2z
f = np.zeros((n + 1) * (2 * n + 1))

# Add the boundary conditions
L[np.arange(0, (n + 1) * (2 * n + 1), n + 1), :] = Dx[
    np.arange(0, (n + 1) * (2 * n + 1), n + 1), :
]  # Neumann x = 1

L[np.arange(n, (n + 1) * (2 * n + 1), n + 1), :] = Dx[
    np.arange(n, (n + 1) * (2 * n + 1), n + 1), :
]  # Neumann x = -1
f[np.arange(n, (n + 1) * (2 * n + 1), n + 1)] = 1

L[(n + 1) * 2 * n :, :] = Dz[(n + 1) * 2 * n :, :]  # Neumann y = -1

L[: 2 * n + 1, :] = np.eye(2 * n + 1, (n + 1) * (2 * n + 1))  # Dirichlet y = 1
f[: 2 * n + 1] = 1

# Solve the linear system
u = np.linalg.solve(L, f)
uu = u.reshape((2 * n + 1, n + 1))

# Create a finer grid for plotting
x_fine = np.linspace(-1, 1, 21)
y_fine = np.linspace(-2, 2, 41)
xxx, yyy = np.meshgrid(x_fine, y_fine)
uuu = sp.griddata(
    (xx.ravel(), zz.ravel()), uu.ravel(), (xxx, yyy / 2), method="cubic"
)

# Plot the solution
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xxx, yyy, uuu, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_xlim(-1, 1)
ax.set_ylim(-2, 2)

plt.show()

# Error analysis
H = 4
L = 2
lam = (np.arange(1, 101) - 1 / 2) * np.pi / H
u_exact = lambda x, y: 1 - np.sum(  # noqa: E731
    np.sin(lam * H)
    * np.cosh(lam * (x - 1))
    * np.cos(lam * (y + 2))
    / (2 * lam**2 * np.sinh(lam * L))
)
uu_exact = np.vectorize(u_exact)(xx, 2 * zz)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    xx, 2 * zz, abs(uu / uu_exact - 1), rstride=1, cstride=1, cmap="viridis"
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("error")

plt.show()
