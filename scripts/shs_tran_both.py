"""Find the flow of a channel with transverse grooves on both walls."""

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

# Construct the differentiation matrices
n = 10
D, c = cheb(n)

# Construct the grid of points
x = np.concatenate((l / 4 * (c - 1), ((L - l) * c - l - L) / 4))
y = (c + 1) / 2
xx_std, yy_std = np.meshgrid(x, y)

xx = np.vstack((xx_std[:, : n + 1], xx_std[:, n + 1 :]))
yy = np.vstack((yy_std[:, : n + 1], yy_std[:, n + 1 :]))

xf = xx.flatten()
yf = yy.flatten()

# Construct the relevant operators
Dx = np.kron(np.eye(n + 1), D)
Dy = np.kron(D, np.eye(n + 1))

D2 = D @ D
D2x = np.kron(np.eye(n + 1), D2)
D2y = np.kron(D2, np.eye(n + 1))

D2x2y = D2x @ D2y

D4 = D2 @ D2
D4x = np.kron(np.eye(n + 1), D4)
D4y = np.kron(D4, np.eye(n + 1))

L1 = (4 / l) ** 4 * D4x + 2 * (4 / l) ** 2 * 2**2 * D2x2y + 2**4 * D4y
L2 = (
    (4 / (L - l)) ** 4 * D4x
    + 2 * (4 / (L - l)) ** 2 * 2**2 * D2x2y
    + 2**4 * D4y
)

f = np.zeros(2 * (n + 1) ** 2)

# First domain boundary conditions
L1[:: n + 1, :] = D2x[:: n + 1, :]  # Symmetry x = 0, xi = 1
f[: (n + 1) ** 2 : n + 1] = 0
L1[1 :: n + 1, :] = Dy[:: n + 1, :]
f[1 : (n + 1) ** 2 : n + 1] = ((c + 1) ** 2 - 4) / 16

L1[: n + 1, :] = np.eye(n + 1, (n + 1) ** 2)  # Interface y = 1, eta = 1
f[: n + 1] = 0
L1[n + 1 : 2 * (n + 1), :] = D2y[: n + 1, :]
f[n + 1 : 2 * (n + 1)] = 1 / 4

L1[(n + 1) * n :, :] = Dx[(n + 1) * n :, :]  # Symmetry y = 0, eta = -1
f[(n + 1) * n : (n + 1) ** 2] = 0
L1[(n + 1) * (n - 1) : (n + 1) * n, :] = D2y[(n + 1) * n :, :]
f[(n + 1) * (n - 1) : (n + 1) * n] = 0

# Second domain boundary conditions
L2[n :: n + 1, :] = D2x[n :: n + 1, :]  # Symmetry x = -L/2, xi = -1
f[(n + 1) ** 2 + n :: n + 1] = 0
L2[n - 1 :: n + 1, :] = Dy[n :: n + 1, :]
f[(n + 1) ** 2 + n - 1 :: n + 1] = ((c + 1) ** 2 - 4) / 16

L2[: n + 1, :] = np.eye(n + 1, (n + 1) ** 2)  # No-slip y = 1, eta = 1
f[(n + 1) ** 2 : (n + 1) * (n + 2)] = 0
L2[n + 1 : 2 * (n + 1), :] = Dy[: n + 1, :]
f[(n + 1) * (n + 2) : (n + 1) * (n + 3)] = 0

L2[(n + 1) * n :, :] = Dx[(n + 1) * n :, :]  # Symmetry y = 0, eta = -1
f[(n + 1) * (2 * n + 1) :] = 0
L2[(n + 1) * (n - 1) : (n + 1) * n, :] = D2y[(n + 1) * n :, :]
f[(n + 1) * 2 * n : (n + 1) * (2 * n + 1)] = 0

# Add constraints to link the domains
A1 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))
A2 = np.zeros(((n + 1) ** 2, (n + 1) ** 2))

L1[n :: n + 1, :] = (1 / l) * Dx[n :: n + 1, :]
A1[n :: n + 1, :] = -(1 / (L - l)) * Dx[:: n + 1, :]
f[n : (n + 1) ** 2 : n + 1] = 0

L2[:: n + 1, :] = Dy[:: n + 1, :]
A2[:: n + 1, :] = -Dy[n :: n + 1, :]
f[(n + 1) ** 2 :: n + 1] = 0

# L2[1 :: n + 1, :] = 0
# L2[1 :: n + 1, :: n + 1] = np.eye(n + 1)
# A2[1 :: n + 1, :] = 0
# A2[1 :: n + 1, n :: n + 1] = -np.eye(n + 1)
# f[(n + 1) ** 2 + 1 :: n + 1] = 0

# Solve the system
L_complete = np.block([[L1, A1], [A2, L2]])
psi = np.linalg.solve(L_complete, f)

plt.spy(L_complete)
plt.show()

# Convert streamfunction to velocities
u = np.concatenate(
    [2 * Dy @ psi[: (n + 1) ** 2], 2 * Dy @ psi[(n + 1) ** 2 :]]
)
v = np.concatenate(
    [
        -(4 / l) * Dx @ psi[: (n + 1) ** 2],
        -(4 / (L - l)) * Dx @ psi[(n + 1) ** 2 :],
    ]
)

# Reshape to get solution grids
uu = u.reshape((2 * (n + 1), n + 1))
vv = v.reshape((2 * (n + 1), n + 1))
psi = psi.reshape((2 * (n + 1), n + 1))
uu_std = np.hstack((uu[: n + 1, :], uu[n + 1 :, :]))
vv_std = np.hstack((vv[: n + 1, :], vv[n + 1 :, :]))
psi_std = np.hstack((psi[: n + 1, :], psi[n + 1 :, :]))

# Plot the solution
fig, ax = plt.subplots(figsize=(6, 6))
plt.quiver(xx_std, yy_std, uu_std, vv_std, scale_units="xy")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Velocity field")
plt.show()

# Plot u surface
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xx_std, yy_std, uu_std, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u$")

plt.show()

# Plot v surface
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xx_std, yy_std, vv_std, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$v$")

plt.show()

# Plot u surface as in Teo, Khoo
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(uu_std, xx_std, yy_std, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("$u$")
ax.set_ylabel("$x$")
ax.set_zlabel("$y$")

plt.show()

# Plot phi surface
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xx_std, yy_std, psi_std, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$\\phi$")

plt.show()
