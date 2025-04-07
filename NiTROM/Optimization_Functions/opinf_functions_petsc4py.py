import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import resolvent4py as res4py
import math
import time

class pool_class_opinf():
    def __init__(self, comm, r, n_traj, len_traj, poly_comp):
        self.comm = comm
        self.r = r
        self.n_traj = n_traj
        self.len_traj = len_traj
        self.poly_comp = poly_comp
        self.compute_mat_sizes()

    def compute_mat_sizes(self):
        self.n_snap = self.n_traj * self.len_traj
        self.nl = res4py.compute_local_size(self.n_snap)
        self.ns = (self.nl, self.n_snap)

        rp_tot = 0
        for p in self.poly_comp:
            rp_tot += math.comb(self.r + p - 1, p)
        self.rp = rp_tot
        self.rpl = res4py.compute_local_size(self.rp)
        self.rps = (self.rpl, self.rp)


def assemble_P(pool, lambdas):
    P = np.zeros(pool.rp)
    shift = 0
    for count, p in enumerate(pool.poly_comp):
        rp = math.comb(pool.r + p - 1, p)
        P[shift:shift + rp] = lambdas[count]
        shift += rp
    
    P_petsc = PETSc.Mat().createAIJ((pool.rps, pool.rps), comm=pool.comm)
    start, end = P_petsc.getOwnershipRange()
    for i in range(start, end):
        P_petsc.setValue(i, i, P[i])
    P_petsc.assemble()
    return P_petsc

def solve_least_squares_problem_petsc4py(pool, Z, Y_T, W, P):
    A_1 = W.matTransposeMult(Z)
    A = Z.matMult(A_1)
    A.axpy(1, P)

    res4py.petscprint(pool.comm, "Computing SVD of A")
    t = time.time()
    svd = SLEPc.SVD().create(pool.comm)
    svd.setOperators(A)
    svd.setType('scalapack')
    svd.solve()
    t2 = time.time() - t
    res4py.petscprint(pool.comm, f"SVD computation time: {t2:.2f} seconds")

    nconv = svd.getConverged()
    res4py.petscprint(pool.comm, f"Number of converged singular values: {nconv}")

    s_inv = []
    U_vectors = []
    V_vectors = []

    for i in range(nconv):
        u = A.createVecLeft()
        v = A.createVecRight()
        s = svd.getSingularTriplet(i, u, v)
        if s > 1e-12:
            s_inv.append(1.0/s)
            U_vectors.append(u)
            V_vectors.append(v)

    # Form matrices U, S, and V
    U = PETSc.Mat().createAIJ((pool.rps, len(U_vectors)), comm=pool.comm)
    V = PETSc.Mat().createAIJ((pool.rps, len(V_vectors)), comm=pool.comm)
    S = PETSc.Mat().createAIJ((len(s_inv), len(s_inv)), comm=pool.comm)

    row_u_start, row_u_end = U.getOwnershipRange()
    row_s_start, row_s_end = S.getOwnershipRange()
    row_v_start, row_v_end = V.getOwnershipRange()
    kept = 0
    for i in range(len(s_inv)):
        u = U_vectors[i]; s = s_inv[i]; v = V_vectors[i]

        u_arr = u.getArray()
        U.setValues(range(row_u_start,row_u_end), [kept], u_arr)

        v_arr = v.getArray()
        V.setValues(range(row_v_start, row_v_end), [kept], v_arr)

        if row_s_start <= kept < row_s_end:
            S.setValue(kept, kept, s)
        kept += 1

    res4py.petscprint(pool.comm, f"Number of singular values kept: {kept}")

    U.assemble()
    S.assemble()
    V.assemble()

    US = U.matMult(S)
    A_inv = US.matTransposeMult(V)

    U.destroy(); V.destroy(); S.destroy(); US.destroy(); A.destroy()

    M1 = Y_T.transposeMatMult(W)
    M2 = M1.matTransposeMult(Z)
    M = M2.matMult(A_inv)
    M1.destroy(); M2.destroy(); A_inv.destroy()
    
    return M