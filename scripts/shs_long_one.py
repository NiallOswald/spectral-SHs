"""Find the flow of a channel with longitudinal grooves on a single wall."""

import numpy as np
from spectral_shs import cheb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Setting parameters
H = 2
E = 2
e = 1

L = E / H
l = e / H  # noqa: E741
delta = e / E

# Construct the differentiation matrices
n = 10
D, c = cheb(n)
Dx = np.kron(np.eye(n + 1), D)
Dy = np.kron(D, np.eye(n + 1))

# Construct the grid of points
x = np.concatenate((l / 4 * (c - 1), ((L - l) * c - l - L) / 4))
y = (c + 1) / 2
xx_std, yy_std = np.meshgrid(x, y)

xx = np.vstack((xx_std[:, : n + 1], xx_std[:, n + 1 :]))
yy = np.vstack((yy_std[:, : n + 1], yy_std[:, n + 1 :]))

xf = xx.flatten()
yf = yy.flatten()

# Construct the operators for each domain
D2 = D @ D
D2x = np.kron(np.eye(n + 1), D2)
D2y = np.kron(D2, np.eye(n + 1))
L1 = 16 / l**2 * D2x + 4 * D2y
L2 = 16 / (L - l) ** 2 * D2x + 4 * D2y
f = -np.ones(2 * (n + 1) ** 2)

# First domain boundary conditions
L1[: n + 1, :] = Dy[: n + 1, :]  # Interface y = 1, eta = 1
f[: n + 1] = 0

L1[(n + 1) * n :, :] = np.zeros(
    (n + 1, (n + 1) ** 2)
)  # No-slip y = 0, eta = -1
L1[(n + 1) * n :, (n + 1) * n :] = np.eye(n + 1)
f[(n + 1) * n : (n + 1) ** 2] = 0

L1[: (n + 1) ** 2 : n + 1, :] = Dx[
    : (n + 1) ** 2 : n + 1, :
]  # Symmetry x = 0, xi = 1
f[: (n + 1) ** 2 : n + 1] = 0

# Second domain boundary conditions
L2[: n + 1, :] = np.eye(n + 1, (n + 1) ** 2)  # No-slip y = 1, eta = 1
f[(n + 1) ** 2 : (n + 1) * (n + 2)] = 0

L2[(n + 1) * n :, :] = np.zeros(
    (n + 1, (n + 1) ** 2)
)  # No-slip y = 0, eta = -1
L2[(n + 1) * n :, (n + 1) * n :] = np.eye(n + 1)
f[(n + 1) * (2 * n + 1) :] = 0

L2[n : (n + 1) ** 2 : n + 1, :] = Dx[
    n : (n + 1) ** 2 : n + 1, :
]  # Symmetry x = -L/2, xi = -1
f[(n + 1) ** 2 + n :: n + 1] = 0

# Add constraints to link the domains
L1[n : (n + 1) ** 2 : n + 1, :] = Dx[n : (n + 1) ** 2 : n + 1, :]
A1 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
A1[n : (n + 1) ** 2 : n + 1, :] = -Dx[: (n + 1) ** 2 : n + 1, :]
f[n : (n + 1) ** 2 : n + 1] = 0

L2[: (n + 1) ** 2 : n + 1, :] = np.zeros((n + 1, (n + 1) ** 2))
L2[: (n + 1) ** 2 : n + 1, : (n + 1) ** 2 : n + 1] = np.eye(n + 1)
A2 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
A2[: (n + 1) ** 2 : n + 1, n : (n + 1) ** 2 : n + 1] = -np.eye(n + 1)
f[(n + 1) ** 2 : 2 * (n + 1) ** 2 : n + 1] = 0

L = np.block([[L1, A1], [A2, L2]])

# Solve the linear system
u = np.linalg.solve(L, f)
uu = u.reshape((2 * (n + 1), n + 1))
uu_std = np.hstack((uu[: n + 1, :], uu[n + 1 :, :]))

# Plot the solution
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xx_std, yy_std, uu_std, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u$")
# ax.set_xlim(-L / 2, 0)
# ax.set_ylim(0, 1)

plt.show()
