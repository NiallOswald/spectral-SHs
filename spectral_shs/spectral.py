"""Module implementing spectral methods."""

import numpy as np


def cheb(n):
    """Return the Chebyshev differentiation matrix of order n."""
    if n == 0:
        return 0, 1

    x = np.array([np.cos(np.pi * np.arange(n + 1) / n)]).T
    c = np.array(
        [np.concatenate([[2], np.ones(n - 1), [2]]) * (-1) ** np.arange(n + 1)]
    ).T
    X = np.tile(x, (1, n + 1))  # noqa: N806
    dX = X - X.T  # noqa: N806
    D = (c @ (1 / c).T) / (dX + np.eye(n + 1))  # noqa: N806
    D = D - np.diag(np.sum(D.T, axis=0))  # noqa: N806

    return D, x.T[0]


def chebfft(v, real=True):
    """Return the derivative of v at the Chebyshev points using the FFT."""
    n = len(v) - 1
    if n == 0:
        return 0

    x = np.cos(np.arange(0, n + 1) * np.pi / n)
    ii = np.arange(0, n)
    V = np.concatenate([v, v[n - 1 : 0 : -1]])  # noqa: N806

    U = np.fft.fft(V)  # noqa: N806
    if real:
        U = U.real  # noqa: N806
    W = np.fft.ifft(  # noqa: N806
        1j * np.concatenate([ii, [0], np.arange(1 - n, 0)]) * U
    )
    if real:
        W = W.real  # noqa: N806

    w = np.zeros(n + 1)
    w[1:n] = -W[1:n] / np.sqrt(1 - x[1:n] ** 2)
    w[0] = np.sum(ii**2 * U[ii]) / n + 0.5 * n * U[n]
    w[n] = (
        np.sum((-1) ** (ii + 1) * ii**2 * U[ii]) / n
        + 0.5 * (-1) ** (n + 1) * n * U[n]
    )

    return w
