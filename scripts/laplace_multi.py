"""Solving a Laplace problem by dividing the domain."""

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
y = np.concatenate((x + 1, x - 1))
xx, yy = np.meshgrid(x, y)

# Construct the Laplace operators for each domain
D2 = D @ D
D2x = np.kron(np.eye(n + 1), D2)
D2y = np.kron(D2, np.eye(n + 1))
L1 = D2x + D2y
L2 = L1.copy()
f = np.zeros(2 * (n + 1) ** 2)

# Add the boundary conditions
L1[np.arange(0, (n + 1) ** 2, n + 1), :] = Dx[
    np.arange(0, (n + 1) ** 2, n + 1), :
]  # Neumann x = 1
L2[np.arange(0, (n + 1) ** 2, n + 1), :] = Dx[
    np.arange(0, (n + 1) ** 2, n + 1), :
]  # Neumann x = 1

L1[np.arange(n, (n + 1) ** 2, n + 1), :] = Dx[
    np.arange(n, (n + 1) ** 2, n + 1), :
]  # Neumann x = -1
L2[np.arange(n, (n + 1) ** 2, n + 1), :] = Dx[
    np.arange(n, (n + 1) ** 2, n + 1), :
]  # Neumann x = -1
f[np.arange(n, 2 * (n + 1) ** 2, n + 1)] = 1

L2[(n + 1) * n :, :] = Dy[(n + 1) * n :, :]  # Neumann y = -2

L1[: n + 1, :] = np.eye(n + 1, (n + 1) ** 2)  # Dirichlet y = 2
f[: n + 1] = 1

# Add constraints to link the domains
L1[(n + 1) * n :, :] = Dy[(n + 1) * n :, :]
A1 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
A1[(n + 1) * n :, :] = -Dy[: n + 1, :]
f[(n + 1) * n : (n + 1) ** 2] = 0

L2[: n + 1, :] = np.eye(n + 1, (n + 1) ** 2)
A2 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
A2[: n + 1, (n + 1) * n :] = -np.eye(n + 1, n + 1)
f[(n + 1) ** 2 : (n + 1) * (n + 2)] = 0

L = np.hstack((np.vstack((L1, A2)), np.vstack((A1, L2))))

# Solve the linear system
u = np.linalg.solve(L, f)
uu = u.reshape((2 * (n + 1), n + 1))

# Create a finer grid for plotting
x_fine = np.linspace(-1, 1, 21)
y_fine = np.linspace(-2, 2, 41)
xxx, yyy = np.meshgrid(x_fine, y_fine)
uuu = sp.griddata(
    (xx.ravel(), yy.ravel()), uu.ravel(), (xxx, yyy), method="cubic"
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
uu_exact = np.vectorize(u_exact)(xx, yy)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    xx, yy, abs(uu - uu_exact), rstride=1, cstride=1, cmap="viridis"
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("error")

plt.show()
