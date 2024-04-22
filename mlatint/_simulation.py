"""
The FDFD code here is a stripped-down version of https://github.com/fancompute/fdfdpy
Besides some minor modifications this is simply a wrapper that adds a PyTorch-
compatible adjoint simulation.
"""

from functools import partial
from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from scipy.constants import epsilon_0, mu_0
from scipy.sparse.linalg import spsolve


class fdfd2d_torch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        eps: torch.Tensor,
        src: NDArray,
        omega: float,
        dl: float,
        npml: tuple[int, int],
        l0: float,
    ) -> torch.Tensor:
        _eps = eps.numpy(force=True)

        A = construct_A(_eps, omega, dl, npml, l0)
        b = 1j * omega * src

        ez_fw = spsolve(A, b.ravel())

        ctx.A = A
        ctx.omega = omega
        ctx.l0 = l0
        ctx.ez_fw = ez_fw

        return torch.as_tensor(
            ez_fw.reshape(eps.shape), dtype=torch.cfloat, device=eps.device
        )

    @staticmethod
    def backward(ctx: Any, dJdE: torch.Tensor) -> tuple:
        _dJdE = dJdE.numpy(force=True)
        ez_aj = spsolve(ctx.A.T, _dJdE.ravel())
        gd = -(ctx.omega**2) * epsilon_0 * ctx.l0 * np.real(ctx.ez_fw * ez_aj)

        return (
            torch.as_tensor(
                gd.reshape(dJdE.shape), dtype=torch.float, device=dJdE.device
            ),
            None,
            None,
            None,
            None,
            None,
        )


def sig_w(l, dw, m=3, lnR=-30):
    sig_max = -(m + 1) * lnR / (2 * np.sqrt(mu_0 / epsilon_0) * dw)
    return sig_max * (l / dw) ** m


def s_value(l, dw, omega, l0):
    return 1 - 1j * sig_w(l, dw) / (omega * epsilon_0 * l0)


def create_sfactor(wrange, l0, s, omega, Nw, Nw_pml):
    sfactor_array = np.ones(Nw, dtype=np.complex128)
    if Nw_pml < 1:
        return sfactor_array
    hw = np.diff(wrange)[0] / Nw
    dw = Nw_pml * hw
    for i in range(0, Nw):
        if s == "f":
            if i <= Nw_pml:
                sfactor_array[i] = s_value(hw * (Nw_pml - i + 0.5), dw, omega, l0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = s_value(
                    hw * (i - (Nw - Nw_pml) - 0.5), dw, omega, l0
                )
        if s == "b":
            if i <= Nw_pml:
                sfactor_array[i] = s_value(hw * (Nw_pml - i + 1), dw, omega, l0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = s_value(hw * (i - (Nw - Nw_pml) - 1), dw, omega, l0)
    return sfactor_array


def S_create(omega, l0, N, npml, xrange, yrange=None, matrix_format="csr"):
    if np.isscalar(npml):
        npml = np.array([npml])
    if len(N) < 2:
        N = np.append(N, 1)
        npml = np.append(npml, 0)
    nx = N[0]
    nx_pml = npml[0]
    ny = N[1]
    ny_pml = npml[1]

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor(xrange, l0, "f", omega, nx, nx_pml)
    s_vector_x_b = create_sfactor(xrange, l0, "b", omega, nx, nx_pml)
    s_vector_y_f = create_sfactor(yrange, l0, "f", omega, ny, ny_pml)
    s_vector_y_b = create_sfactor(yrange, l0, "b", omega, ny, ny_pml)

    Sx_f = np.repeat(1 / s_vector_x_f[:, None], ny, axis=1)
    Sx_b = np.repeat(1 / s_vector_x_b[:, None], ny, axis=1)
    Sy_f = np.repeat(1 / s_vector_y_f[None, :], nx, axis=0)
    Sy_b = np.repeat(1 / s_vector_y_b[None, :], nx, axis=0)

    # Construct the 1D total s-array into a diagonal matrix
    Sx_f = sp.diags(Sx_f.ravel(), format=matrix_format)
    Sx_b = sp.diags(Sx_b.ravel(), format=matrix_format)
    Sy_f = sp.diags(Sy_f.ravel(), format=matrix_format)
    Sy_b = sp.diags(Sy_b.ravel(), format=matrix_format)

    return Sx_f, Sx_b, Sy_f, Sy_b


def construct_A(
    eps_r,
    omega,
    dl,
    npml,
    l0,
    matrix_format="csr",
):
    N = np.asarray(eps_r.shape)
    xrange = [0, N[0] * dl]
    yrange = [0, N[1] * dl]

    eps_vec = epsilon_0 * l0 * eps_r.ravel()
    eps = sp.diags(eps_vec, format=matrix_format)

    Sxf, Sxb, Syf, Syb = S_create(
        omega, l0, N, npml, xrange, yrange, matrix_format=matrix_format
    )

    dl = np.array([np.diff(xrange)[0], np.diff(yrange)[0]]) / N
    Dxb = Sxb.dot(createDws("x", "b", dl, N, matrix_format=matrix_format))
    Dxf = Sxf.dot(createDws("x", "f", dl, N, matrix_format=matrix_format))
    Dyb = Syb.dot(createDws("y", "b", dl, N, matrix_format=matrix_format))
    Dyf = Syf.dot(createDws("y", "f", dl, N, matrix_format=matrix_format))

    A = (Dxf / (mu_0 * l0)) @ Dxb + (Dyf / (mu_0 * l0)) @ Dyb + omega**2 * eps

    return A


def createDws(w, s, dl, N, bloch=(0, 0), matrix_format="csr"):
    nx = N[0]
    dx = dl[0]
    if len(N) != 1:
        ny = N[1]
        dy = dl[1]
    else:
        ny = 1
        dy = np.inf
    if w == "x":
        if s == "f":
            dxf = sp.diags([-1, 1, 1], [0, 1, -nx + 1], shape=(nx, nx))
            Dws = 1 / dx * sp.kron(dxf, sp.eye(ny), format=matrix_format)
        else:
            dxb = sp.diags([1, -1, -1], [0, -1, nx - 1], shape=(nx, nx))
            Dws = 1 / dx * sp.kron(dxb, sp.eye(ny), format=matrix_format)
    if w == "y":
        if s == "f":
            dyf = sp.diags([-1, 1, 1], [0, 1, -ny + 1], shape=(ny, ny))
            Dws = 1 / dy * sp.kron(sp.eye(nx), dyf, format=matrix_format)
        else:
            dyb = sp.diags([1, -1, -1], [0, -1, ny - 1], shape=(ny, ny))
            Dws = 1 / dy * sp.kron(sp.eye(nx), dyb, format=matrix_format)
    return Dws
