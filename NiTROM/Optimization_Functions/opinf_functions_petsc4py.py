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
    
    p_petsc = PETSc.Vec().createWithArray(P, pool.rps, comm=pool.comm)
    p_petsc.setType('mpi')
    return p_petsc


def solve_least_squares_problem_petsc4py(pool, Z, Y_T, w, p):
    # Z.transpose()
    # A_1 = Z.copy()
    # Z.transpose()
    # A_1.diagonalScale(w, None)
    # A = Z.matMult(A_1)
    # A_1.destroy()
    # A.setDiagonal(p, addv=PETSc.InsertMode.ADD_VALUES)
    # p.destroy()

    # res4py.petscprint(pool.comm, "Computing SVD of A")
    # t = time.time()
    # svd = SLEPc.SVD().create(pool.comm)
    # svd.setOperators(A)
    # svd.setType('scalapack')
    # svd.solve()
    # t2 = time.time() - t
    # res4py.petscprint(pool.comm, f"SVD computation time: {t2:.2 f} seconds")

    # nconv = svd.getConverged()
    # res4py.petscprint(pool.comm, f"Number of converged singular values: {nconv}")

    # s_inv = []
    # U_vectors = []
    # V_vectors = []

    # for i in range(nconv):
    #     u = A.createVecLeft()
    #     v = A.createVecRight()
    #     s = svd.getSingularTriplet(i, u, v)
    #     if s > 1e-12:
    #         s_inv.append(1.0/s)
    #         U_vectors.append(u)
    #         V_vectors.append(v)
    # svd.destroy()

    # # Form matrices U, S, and V
    # U = PETSc.Mat().createDense((pool.rps, len(U_vectors)), comm=pool.comm)
    # V = PETSc.Mat().createDense((pool.rps, len(V_vectors)), comm=pool.comm)
    # S = PETSc.Vec().create(len(s_inv), comm=pool.comm)
    # S.setType('seq')

    # row_u_start, row_u_end = U.getOwnershipRange()
    # row_v_start, row_v_end = V.getOwnershipRange()
    # for i in range(len(s_inv)):
    #     u = U_vectors[i]; s = s_inv[i]; v = V_vectors[i]

    #     u_arr = u.getArray()
    #     U.setValues(range(row_u_start,row_u_end), [i], u_arr)

    #     v_arr = v.getArray()
    #     V.setValues(range(row_v_start, row_v_end), [i], v_arr)

    #     S.setValue(i, s)

    # res4py.petscprint(pool.comm, f"Number of singular values kept: {len(s_inv)}")

    # U.assemble()
    # V.assemble()

    # U.diagonalScale(None, S)
    # A_inv = U.matTransposeMult(V)

    # U.destroy(); V.destroy(); S.destroy(); A.destroy()

    P = PETSc.Mat().createAIJ((pool.rps, pool.rps), comm=pool.comm)
    P.setDiagonal(p)
    P.assemble()
    ksp = res4py.create_mumps_solver(pool.comm, P)
    A = res4py.MatrixLinearOperator(pool.comm, P, ksp)
    # B = ZW, K = I, C = Z
    B_mat = Z.copy()
    B_mat.diagonalScale(None, w)
    B = SLEPc.BV().createFromMat(B_mat)
    B.setType('mat')
    B_mat.destroy()
    K = np.identity(pool.n_snap)
    C = SLEPc.BV().createFromMat(Z)
    C.setType('mat')
    
    L = res4py.LowRankUpdatedLinearOperator(pool.comm, A, B, K, C)
    res4py.petscprint(pool.comm, "Computing SVD ...")
    U, S, V = res4py.randomized_svd(L, L.solve_mat, 100*10, 2, 100)
    print(1/S.diagonal())

    Y_T.transpose()
    Y_T.diagonalScale(None, w)
    M1 = Y_T.matTransposeMult(Z)
    M = M1.matMult(A_inv)
    M1.destroy(); A_inv.destroy()
    
    return M