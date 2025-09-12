import numpy as np
import scipy as sp

def opinf_ep(mpi_pool, phi, lambdas):
    # Assemble and preallocate matrices for OpInf EP
    m = mpi_pool.n_snapshots*mpi_pool.n_traj
    r = phi.shape[-1]
    Z = np.zeros((r, m))
    dZ = np.zeros((r, m))
    W = np.zeros(m)
    for i in range(mpi_pool.n_traj):
        Z[:, i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = phi.T @ mpi_pool.X[i, :, :]
        dZ[:, i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = phi.T @ mpi_pool.dX[i, :, :]
        W[i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = 1./(mpi_pool.weights[i]*mpi_pool.n_traj)
    W = np.diag(W)
    A = np.zeros((r, r))
    H = np.zeros((r, r*r))

    vkronf = np.empty((r*r, m))
    for k in range(m):
        z = Z[:, k]
        vkronf[:, k] = np.kron(z, z)
    
    # Iterate through rows to compute parts of A and H
    for i in range(r):
        F = dZ - H @ vkronf  # Remove known quadratic part

        # Build vkron of remaining pairs
        rows = []
        for k in range(m):
            z = Z[:, k]
            rows.append(np.concatenate([z[j] * z[i+1:r] for j in range(r)]))
        vkron = np.column_stack(rows)
        qv = vkron.shape[0]
        D = np.vstack([Z, vkron])
        lhs = D @ W @ D.T
        rhs = D @ W @ F[i, :].T
        reg = np.eye(r + qv)
        reg[:r, :r] *= lambdas[0]
        reg[r:, r:] *= lambdas[1]

        G = sp.linalg.solve(lhs + reg, rhs)
        A[i, :] = G[:r]
        offset = r
        # Fill inferred part of H from quadratic part of G
        for j in range(r):
            zstart = r * j
            zcount = j * (r - i - 1)
            cnt = r - i - 1
            if cnt > 0:
                H[i, (i+1) + zstart : (i+1) + zstart + cnt] = G[offset + zcount : offset + zcount + cnt]
        # Enforce skew-symmetry
        for j in range(r):
            zstart = r * j
            H[i:r, zstart + i] = -H[i, zstart + i : zstart + r]
        
    return A, H.reshape(r,r,r)