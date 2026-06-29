import numpy as np
import torch
from string import ascii_lowercase as ascii

import time as tlib
from ..PyTorch_Functions.integrators import my_etdrk4, etdrk4_setup, my_rk4
from ..PyTorch_Functions.linear_interpolation import Interp1D
from .rom_utils import construct_operators, propagate_gradients


from ..time_steppers.time_stepper import solve_ivp


class NiTROMCostAndGrad:
    
    def __init__(self, opt_obj, rom_model):
        self.opt_obj = opt_obj
        self.rom_model = rom_model

    def cost(self, params):
        r"""
        Evaluate cost function of the form

        .. math::

            J = \sum_{j=0}^{N_{\text{traj}}-1}\frac{1}{\alpha_j}
                \sum_{i=0}^{N-1}\lVert y^{(j)}(t_i) - \hat{y}^{(j)}(t_i)\rVert^2

        
        """
        latent_space_model = self.rom_model.model
        self.rom_model.update(params)
        Z = solve_ivp(
            latent_space_model.evaluate_rhs,
            x0=self.rom_model.projection.encode(self.opt_obj.X[:, :, 0]),
            t0=self.opt_obj.time[0],
            tf=self.opt_obj.time[-1],
            dt=(self.opt_obj.time[1] - self.opt_obj.time[0]) / 100,
            t_eval=self.opt_obj.time,
            method="rk4",
            external_forcing=self.opt_obj.forcing_fns,
        )
        Yhat = latent_space_model.evaluate_output(self.rom_model.projection.decode(Z))
        Y = latent_space_model.evaluate_output(self.opt_obj.X)
        e = Y - Yhat
        J = torch.sum((e * e).sum(dim=(1, 2)) / self.opt_obj.weights)
        return J

    def gradient(self, params):


        


def create_objective_and_gradient(*args, **kwargs):
    """
    opt_obj:        instance of class "optimization_objects" in file "classes.py"
    pool:           instance of the class "pool" in file "classes.py"
    fom:            instance of the full-order model class
    """

    opt_obj, fom = args
    pool = opt_obj.pool
    euclidean_hessian = None
    glob_stable = kwargs.get("glob_stable", False)
    poly_comp = kwargs.get("poly_comp", None)
    return_numpy = kwargs.get("return_numpy", False)

    integrator = kwargs.get("integrator", my_rk4)
    rk4_substeps = int(kwargs.get("rk4_substeps", 5))
    etdrk4_substeps = int(kwargs.get("etdrk4_substeps", 5))  # used by ETDRK4 branch

    def cost(*params):
        """
        Evaluate the cost function
        Phi and Psi:    bases (size N x r) that define the projection operator
        tensors:        (A2,A3,...)
        """
        Phi, Psi = params[0], params[1]
        tensors_old = params[2:]
        if glob_stable:
            tensors, _ = construct_operators(tensors_old, poly_comp)
        else:
            tensors = tensors_old

        PhiF = Phi @ torch.linalg.inv(Psi.T @ Phi)

        dt = opt_obj.time[1] - opt_obj.time[0]
        if integrator == my_etdrk4:
            dt /= etdrk4_substeps
            D, V = torch.linalg.eig(tensors[0])
            V_inv = torch.linalg.inv(V)
            linop = V, D, V_inv
            etdrk4_coefs = etdrk4_setup(linop, dt)

        J = torch.zeros((), device=pool.device, dtype=Phi.dtype)
        B = opt_obj.my_n_traj
        if B > 0:
            X0 = opt_obj.X[:, :, 0]  # (B, N)
            z0 = X0 @ Psi  # (B, r)
            u_batch = opt_obj.F.T @ Psi  # (B, r)

            if integrator == my_etdrk4:
                sol = integrator(
                    etdrk4_coefs,
                    opt_obj.evaluate_rom_rhs_nonlinear,
                    opt_obj.time,
                    z0,
                    etdrk4_substeps,
                    args=(u_batch,) + tensors,
                )
            else:
                sol = integrator(
                    opt_obj.evaluate_rom_rhs,
                    opt_obj.time,
                    z0,
                    args=(u_batch,) + tensors,
                    n_substeps=rk4_substeps,
                )

            Y_true = fom.compute_output(opt_obj.X)
            Y_model = fom.compute_output(torch.matmul(PhiF, sol))

            e = Y_true - Y_model
            err_per_traj = (e * e).sum(dim=(1, 2))
            J = J + torch.sum(err_per_traj / opt_obj.weights)  # scalar tensor

        if opt_obj.l2_pen is not None and pool.rank == 0:
            time_pen = torch.linspace(
                0,
                opt_obj.pen_tf,
                opt_obj.n_snapshots * opt_obj.nsave_rom,
                device=pool.device,
            )
            if integrator == my_etdrk4:
                Z = integrator(
                    etdrk4_coefs, lambda t, z: 0 * z, time_pen, opt_obj.randic
                )
            else:
                Z = integrator(
                    opt_obj.evaluate_rom_rhs,
                    opt_obj.time,
                    z0,
                    args=(u_batch,) + tensors,
                )
            J = J + opt_obj.l2_pen * torch.dot(Z[:, -1], Z[:, -1])

        if return_numpy:
            return J.detach().cpu().numpy()

        return J

    def euclidean_gradient(*params):
        """
        Evaluate the euclidean gradient of the cost function with respect to the parameters
        Phi and Psi:    bases (size N x r) that define the projection operator
        tensors:        (A2,A3,...)
        """

        Phi, Psi = params[0], params[1]
        tensors_old = params[2:]
        if glob_stable:
            tensors, other_tensors = construct_operators(tensors_old, poly_comp)
        else:
            tensors = tensors_old

        dt = (opt_obj.time[1] - opt_obj.time[0]) / etdrk4_substeps
        # Keep everything in-model dtype (float64 here will slow GPU kernels / cause promotions)
        t_unit = torch.linspace(
            0.0,
            1.0,
            steps=opt_obj.nsave_rom,
            device=pool.device,
            dtype=Phi.dtype,
        )

        if integrator == my_etdrk4:
            D, V = torch.linalg.eig(tensors[0])
            V_inv = torch.linalg.inv(V)
            linop = V, D, V_inv
            linop_T = V_inv.T, D, V.T
            dt2 = dt / (opt_obj.nsave_rom - 1)
            etdrk4_coefs = etdrk4_setup(linop, dt)
            etdrk4_coefs_2 = etdrk4_setup(linop, dt2)
            etdrk4_coefs_T2 = etdrk4_setup(linop_T, dt2)

        B = opt_obj.my_n_traj

        # Initialize arrays to store the gradients
        n, r = Phi.shape
        grad_Phi = torch.zeros((B, n, r), device=pool.device, dtype=Phi.dtype)
        grad_Psi = torch.zeros((B, n, r), device=pool.device, dtype=Phi.dtype)
        grad_tensors = [
            torch.zeros((B, *tensor.shape), device=pool.device, dtype=Phi.dtype)
            for tensor in tensors
        ]

        # Initialize arrays needed for future computations
        lam_j_0 = torch.zeros((B, r), device=pool.device, dtype=Phi.dtype)
        Int_lambda = torch.zeros((B, r), device=pool.device, dtype=Phi.dtype)

        # Biorthogonalize Phi and Psi
        F = torch.linalg.inv(Psi.T @ Phi)
        PhiF = Phi @ F

        tlg = opt_obj.tlg
        wlg = opt_obj.wlg

        if B > 0:
            X0 = opt_obj.X[:, :, 0]  # (B, N)
            z0 = X0 @ Psi  # (B, r)

            # u_k = Psi.T @ F[:,k] => u = F.T @ Psi -> (B, r)
            u_batch = opt_obj.F.T @ Psi  # (B, r)

            if integrator == my_etdrk4:
                Z = integrator(
                    etdrk4_coefs,
                    opt_obj.evaluate_rom_rhs_nonlinear,
                    opt_obj.time,
                    z0,
                    etdrk4_substeps,
                    args=(u_batch,) + tensors,
                )  # (B, r, T)
            else:
                Z = integrator(
                    opt_obj.evaluate_rom_rhs,
                    opt_obj.time,
                    z0,
                    args=(u_batch,) + tensors,
                    n_substeps=rk4_substeps,
                )

            # X_z = PhiF@Z
            X_z = torch.einsum(
                "jk, ikm -> ijm", PhiF, Z
            )  # (B, n, T) = (n, r) x (B, r, T)
            # Compute outputs in batch
            # y_true = C @ X  -> (B, 1, T); y_model = C @ (PhiF @ Z) -> (B, 1, T)
            Y_true = fom.compute_output(opt_obj.X)  # (B, no, T)
            Y_model = fom.compute_output(torch.matmul(PhiF, Z))  # (B, no, T)

            e = Y_true - Y_model  # (B, no, T)

            Cte = torch.einsum(
                "jk, ikm -> ijm", fom.compute_output_derivative(PhiF @ Z).T, e
            )  # (B, n, T) = (n, no) x (B, no, T) along axis 1
            # PCte = PhiF.T@Cte # (r, N) x ()
            PCte = torch.einsum(
                "jk, ikm -> ijm", PhiF.T, Cte
            )  # (B, r, T) = (r, n) x (B, n, T)
            alpha = opt_obj.weights

            lam_j_0 *= 0.0
            Int_lambda *= 0.0

            # PsiPCte = Psi@PCte
            PsiPCte = torch.einsum(
                "jk, ikm -> ijm", Psi, PCte
            )  # (B, n, T) = (n, r) x (B, r, T)
            C_minus = Cte - PsiPCte
            FZ = torch.einsum("jk, ikm -> ijm", F, Z)  # (B, r, T) = (r, r) x (B, r, T)
            # FZ = F@Z

            # grad_Psi = grad_Psi + (2/alpha) * X_part @ PCte_part.T
            # grad_Phi = -(2/alpha) * C_minus_part @ FZ_part.T
            # grad_Psi.addmm_(X_z, PCte.T, beta=1.0, alpha=1.0) # <-------- Fix these
            # grad_Phi.addmm_(C_minus, FZ.T, beta=1.0, alpha=1.0)
            # print(X_z.shape, PCte.T.shape)
            grad_Psi.add_(torch.einsum("ijk, imk -> ijm", X_z, PCte))
            grad_Phi.add_(torch.einsum("ijk, imk -> ijm", C_minus, FZ))
            grad_Psi.mul_(2 / alpha[:, None, None])
            grad_Phi.mul_(-2 / alpha[:, None, None])
            # grad_Psi *= 2 / alpha[:, None, None]
            # grad_Phi *= - 2 / alpha[:, None, None]

            for j in range(opt_obj.n_snapshots - 1):
                PCtej = PCte[:, :, opt_obj.n_snapshots - j - 1]

                # Compute the sums in (2.13) and (2.14) in the arXiv paper. Notice that this loop sums backwards
                # from j = N-1 to j = 1, so we will compute the term j = 0 after this loop
                # grad_Psi += (2/alpha)*torch.einsum('i,j',x_zj,PCtej)
                # grad_Phi += -(2/alpha)*torch.einsum('i,j',Ctej - Psi@PCtej,F@zj)

                # ------ Compute the fwd ROM solution between times t0_j and tf_j ---------
                id1 = opt_obj.n_snapshots - 1 - j
                id0 = id1 - 1

                tf_j = opt_obj.time[id1]
                t0_j = opt_obj.time[id0]
                z0_j = Z[:, :, id0]

                delta = tf_j - t0_j
                time_rom_j = t0_j + t_unit * delta
                if torch.abs(time_rom_j[-1] - tf_j) >= 1e-6:
                    print(time_rom_j[-1], tf_j)
                    raise ValueError(
                        "Error in euclidean_gradient() - final time is not correct!"
                    )

                # sol_j = my_etdrk4(etdrk4_coefs_2,opt_obj.evaluate_rom_rhs_nonlinear,time_rom_j,z0_j,internal_steps,args=(u,)+tensors)
                # Z_j = torch.fliplr(sol_j)
                # fZ = Interp1D(time_rom_j,Z_j,extrapolate=True)

                if integrator == my_etdrk4:
                    sol_j = integrator(
                        etdrk4_coefs_2,
                        opt_obj.evaluate_rom_rhs_nonlinear,
                        time_rom_j,
                        z0_j,
                        etdrk4_substeps,
                        args=(u_batch,) + tensors,
                    )
                else:
                    sol_j = integrator(
                        opt_obj.evaluate_rom_rhs,
                        time_rom_j,
                        z0_j,
                        args=(u_batch,) + tensors,
                        n_substeps=rk4_substeps,
                    )
                Z_j = torch.flip(sol_j, dims=[-1])
                fZ = Interp1D(time_rom_j, Z_j, extrapolate=True)

                # --------------------------------------------------------------------------

                # ------ Compute the adj ROM solution between times t0_j and tf_j ----------
                # lam_j_0 += (2/alpha)*PCtej
                # sol_lam = my_etdrk4(etdrk4_coefs_T2,opt_obj.evaluate_rom_adjoint_nonlinear,time_rom_j,lam_j_0,internal_steps,args=(fZ,)+tensors)
                # Lam = torch.fliplr(sol_lam)
                # lam_j_0 = Lam[:,0]
                # Z_j = torch.fliplr(Z_j)

                lam_j_0 += 2 * PCtej / alpha[:, None]
                if integrator == my_etdrk4:
                    sol_lam = integrator(
                        etdrk4_coefs_T2,
                        opt_obj.evaluate_rom_adjoint_nonlinear,
                        time_rom_j,
                        lam_j_0,
                        etdrk4_substeps,
                        args=(fZ,) + tensors,
                    )
                else:
                    sol_lam = integrator(
                        opt_obj.evaluate_rom_adjoint,
                        time_rom_j,
                        lam_j_0,
                        args=(fZ,) + tensors,
                        n_substeps=rk4_substeps,
                    )
                Lam = torch.flip(sol_lam, dims=[-1])
                lam_j_0 = Lam[:, :, 0]
                Z_j = torch.flip(Z_j, dims=[-1])

                # --------------------------------------------------------------------------

                # Interpolate Z_j and Lam onto Gauss-Legendre points
                a = (tf_j - t0_j) / 2
                b = (tf_j + t0_j) / 2
                time_j_lg = a * tlg + b

                fZ = Interp1D(time_rom_j, Z_j, extrapolate=True)
                fL = Interp1D(time_rom_j, Lam, extrapolate=True)
                Z_j_lg = fZ(time_j_lg)  # (B, r, tlg)
                Lam_lg = fL(time_j_lg)  # (B, r, tlg)

                Int_lambda.add_(a * torch.einsum("ijk, k -> ij", Lam_lg, wlg))
                for count, p in enumerate(opt_obj.poly_comp):
                    equation = (
                        "k"
                        + "i,k".join(ascii[: p + 1])
                        + "i,i -> k"
                        + "".join(ascii[: p + 1])
                    )
                    operands = [Lam_lg] + [Z_j_lg for _ in range(p)] + [wlg]
                    grad_tensors[count].add_(-a * torch.einsum(equation, *operands))

            # Add the contribution of the initial condition (last term in (2.14)) to grad_Psi.
            # Add also the contribution of the steady forcing to grad_Psi.
            # x0 = opt_obj.X[k,:,0]
            # f_k = opt_obj.F[:,k]
            # grad_Psi.add_(-torch.outer(x0, lam_j_0))
            # grad_Psi.add_(-torch.outer(f_k, Int_lambda))
            grad_Psi.add_(
                -torch.einsum("ik, ij -> ikj", opt_obj.X[:, :, 0], lam_j_0)
            )  # (B, n, r) = (B, n) x (B, r))
            grad_Psi.add_(
                -torch.einsum("ik, ij -> ikj", opt_obj.F.T, Int_lambda)
            )  # (B, n, r) = (B, n) x (B, r))

        # Compute the gradient of the stability-promoting term
        if opt_obj.l2_pen is not None and pool.rank == 0:
            print("Inside stability-promoting penalty")
            idx = opt_obj.poly_comp.index(1)  # index of the linear tensor

            time_pen = torch.linspace(
                0,
                opt_obj.pen_tf,
                opt_obj.n_snapshots * opt_obj.nsave_rom,
                device=pool.device,
            )
            Z = my_etdrk4(etdrk4_coefs, lambda t, z: 0 * z, time_pen, opt_obj.randic)
            Mu = my_etdrk4(
                etdrk4_coefs,
                lambda t, z: 0 * z,
                time_pen,
                -2 * opt_obj.l2_pen * Z[:, -1],
            )
            Mu = torch.fliplr(Mu)

            for k in range(opt_obj.n_snapshots - 1):
                k0, k1 = k * opt_obj.nsave_rom, (k + 1) * opt_obj.nsave_rom
                fZ = Interp1D(time_pen[k0:k1], Z[:, k0:k1], extrapolate=True)
                fMu = Interp1D(time_pen[k0:k1], Mu[:, k0:k1], extrapolate=True)

                a = (time_pen[k1 - 1] - time_pen[k0]) / 2
                b = (time_pen[k1 - 1] + time_pen[k0]) / 2
                time_k_lg = a * tlg + b

                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)

                for i in range(opt_obj.leggauss_deg):
                    grad_tensors[idx] += (
                        -a * wlg[i] * torch.einsum("i,j", Muk[:, i], Zk[:, i])
                    )

        if opt_obj.which_fix == "fix_bases":
            grad_Phi *= 0.0
            grad_Psi *= 0.0
        elif opt_obj.which_fix == "fix_tensors":
            for k in range(len(grad_tensors)):
                grad_tensors[k] *= 0.0

        grad_Phi = torch.sum(grad_Phi, dim=0)  # (n, r)
        grad_Psi = torch.sum(grad_Psi, dim=0)
        for i in range(len(grad_tensors)):
            grad_tensors[i] = torch.sum(grad_tensors[i], dim=0)

        if glob_stable:
            grad_tensors_new = propagate_gradients(
                grad_tensors, other_tensors, tensors_old, poly_comp
            )
        else:
            grad_tensors_new = grad_tensors

        if return_numpy:
            grad_Phi = grad_Phi.cpu().numpy()
            grad_Psi = grad_Psi.cpu().numpy()
            grad_tensors_new = tuple(
                tensor.cpu().numpy() for tensor in grad_tensors_new
            )
            return grad_Phi, grad_Psi, *grad_tensors_new

        return grad_Phi, grad_Psi, *grad_tensors_new

    return cost, euclidean_gradient, euclidean_hessian
