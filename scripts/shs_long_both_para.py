"""Channel with longitudinal grooves on both walls and parabolic meniscus."""

import numpy as np
from spectral_shs import cheb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Setting parameters
H = 1
E = 2
e = 1.5

L = E / H
l = e / H  # noqa: E741
delta = e / E

theta = 3 * np.pi / 4
mu = theta - np.pi / 2

# Construct the differentiation matrix
n = 10
D, c = cheb(n)

# Construct the grid of points
alpha = (l * (1 / 4 - (c - 1) ** 2 / 16) * np.tan(mu) + 1) ** -1
alphaf = np.tile(alpha, n + 1)

x1 = l / 4 * (c - 1)
x2 = ((L - l) * c - l - L) / 4

x = np.concatenate((x1, x2))

y2 = (c + 1) / 2

xx1 = np.tile(x1, (n + 1, 1))
yy1 = (c.reshape(n + 1, 1) + 1) / 2 @ (alpha**-1).reshape(1, n + 1)
xx2, yy2 = np.meshgrid(x2, y2)

xx_std = np.hstack((xx1, xx2))
yy_std = np.hstack((yy1, yy2))

xx = np.vstack((xx1, xx2))
yy = np.vstack((yy1, yy2))

x1f = xx1.flatten()
y1f = yy1.flatten()
x2f = xx2.flatten()
y2f = yy2.flatten()

xf = xx.flatten()
yf = yy.flatten()

# Construct the operators for each domain
Dx = np.kron(np.eye(n + 1), D)
Dy = np.kron(D, np.eye(n + 1))

Dxy = Dx @ Dy

D2 = D @ D
D2x = np.kron(np.eye(n + 1), D2)
D2y = np.kron(D2, np.eye(n + 1))

L1 = (
    16 / l**2 * D2x
    + (4 / l) * np.tan(mu) * np.diag((x1f - 1) * (y1f + 1) * alphaf) @ Dxy
    + (np.tan(mu) ** 2 / 2)
    * np.diag((x1f - 1) ** 2 * (y1f + 1) * alphaf**2)
    @ D2y
    + (2 / l) * np.tan(mu) * np.diag((y1f + 1) * alphaf) @ Dy
    + (np.tan(mu) ** 2 / 4)
    * np.diag((x1f - 1) ** 2 * (y1f + 1) ** 2 * alphaf**2)
    @ D2y
    + 4 * np.diag(alphaf**2) @ D2y
)
L2 = 16 / (L - l) ** 2 * D2x + 4 * D2y

f1 = -np.ones((n + 1) ** 2)
f2 = -np.ones((n + 1) ** 2)

# First domain boundary conditions
L1[: n + 1, :] = Dy[: n + 1, :]  # Interface y = 1, eta = 1
f1[: n + 1] = 0

L1[(n + 1) * n :, :] = Dy[(n + 1) * n :, :]  # Symmetry y = 0, eta = -1
f1[(n + 1) * n :] = 0

L1[:: n + 1, :] = Dx[:: n + 1, :]  # Symmetry x = 0, xi = 1
f1[:: n + 1] = 0

# Second domain boundary conditions
L2[: n + 1, :] = np.eye(n + 1, (n + 1) ** 2)  # No-slip y = 1, eta = 1
f2[: n + 1] = 0

L2[(n + 1) * n :, :] = Dy[(n + 1) * n :, :]  # Symmetry y = 0, eta = -1
f2[(n + 1) * n :] = 0

L2[n :: n + 1, :] = Dx[n :: n + 1, :]  # Symmetry x = -L/2, xi = -1
f2[n :: n + 1] = 0

# Add constraints to link the domains
L1[n :: n + 1, :] = (
    np.tan(mu) * np.diag(c + 1) @ Dy[n :: n + 1, :]
    - (4 / l) * Dx[n :: n + 1, :]
)
A1 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
A1[n :: n + 1, :] = (4 / (L - l)) * Dx[:: n + 1, :]
f1[n :: n + 1] = 0

L2[:: n + 1, :] = 0
L2[:: n + 1, :: n + 1] = np.eye(n + 1)
A2 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
A2[:: n + 1, n :: n + 1] = -np.eye(n + 1)
f2[:: n + 1] = 0

L_complete = np.block([[L1, A1], [A2, L2]])
f = np.concatenate([f1, f2])

# Solve the linear system
u = np.linalg.solve(L_complete, f)
uu = u.reshape((2 * (n + 1), n + 1))
uu_std = np.hstack((uu[: n + 1, :], uu[n + 1 :, :]))

# Plot the solution
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xx_std, yy_std, uu_std, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u$")

plt.show()
