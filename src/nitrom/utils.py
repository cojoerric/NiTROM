import torch
import torch.distributed as dist


def compute_POD(
    pool,
    normalize: bool = False,
    broadcast: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    r"""
    Compute the proper orthogonal decomposition (POD) of the training data.

    Each rank reshapes its local trajectory snapshots
    (``pool.X`` of shape ``(my_n_traj, N, n_snapshots)``) into a
    ``(N, my_n_traj * n_snapshots)`` matrix.  These are gathered onto the
    root rank (rank 0) and concatenated, in rank order, into the global
    snapshot matrix

    .. math::

        X \in \mathbb{R}^{N \times (n_\text{traj} \cdot n_\text{snaps})},

    on which the economy SVD :math:`X = U\,\mathrm{diag}(S)\,V^\top` is
    computed.  The columns of :math:`U` are the POD modes.

    :param pool: training-data pool holding the (possibly rank-distributed)
        trajectories in ``pool.X``
    :type pool: TrainingPool
    :param normalize: if ``True``, scale each trajectory ``k`` by
        :math:`1/\sqrt{w_k}` (``pool.weights[k]``) before assembling the
        snapshot matrix, so each trajectory contributes to the POD in
        inverse proportion to its weight
    :type normalize: bool
    :param broadcast: if ``True`` (default), the modes ``U`` and singular
        values ``S`` are broadcast from root to **every** rank, so the basis
        is directly usable everywhere; the temporal coefficients ``V`` stay on
        root.  If ``False``, all factors live only on root.
    :type broadcast: bool
    :returns: the economy SVD ``(U, S, V)`` -- ``U`` of shape ``(N, k)``,
        ``S`` of shape ``(k,)``, ``V`` of shape ``(M, k)`` with
        ``k = min(N, M)`` and ``M = n_traj * n_snapshots``.  In a distributed
        run, ``V`` is ``None`` off root; ``U`` and ``S`` are also ``None`` off
        root unless ``broadcast`` is ``True``.
    :rtype: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    """
    X = pool.X
    if normalize:
        # Scale each trajectory by 1 / sqrt(weight).
        X = X / torch.sqrt(pool.weights).reshape(-1, 1, 1)

    # Local snapshot matrix: (N, my_n_traj * n_snapshots).
    X_local = X.permute(1, 0, 2).reshape(pool.N, -1)

    distributed = (
        pool.world_size > 1 and dist.is_available() and dist.is_initialized()
    )

    if not distributed:
        U, S, Vh = torch.linalg.svd(X_local, full_matrices=False)
        return U, S, Vh.mH

    # Gather each rank's local matrix onto root (sizes differ across ranks
    # when n_traj is not divisible by world_size) and compute the SVD there.
    gather_list = [None] * pool.world_size if pool.rank == 0 else None
    dist.gather_object(X_local, gather_list, dst=0)
    if pool.rank == 0:
        X = torch.cat(
            [g.to(device=pool.device, dtype=pool.dtype) for g in gather_list],
            dim=1,
        )
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.mH
    else:
        U, S, V = None, None, None

    if broadcast:
        # Distribute the modes and singular values to every rank.
        k = min(pool.N, pool.n_traj * pool.n_snapshots)
        if pool.rank == 0:
            U, S = U.contiguous(), S.contiguous()
        else:
            U = torch.empty((pool.N, k), device=pool.device, dtype=pool.dtype)
            S = torch.empty((k,), device=pool.device, dtype=pool.dtype)
        dist.broadcast(U, src=0)
        dist.broadcast(S, src=0)

    return U, S, V


def interp_quadratic(
    t_eval: torch.Tensor,
    t_data: torch.Tensor,
    y_data: torch.Tensor,
) -> torch.Tensor:
    r"""
    Piecewise quadratic (3-point Lagrange) interpolation of uniformly
    sampled data.

    For each query point in *t_eval*, the three nearest data points are
    used to build a degree-2 Lagrange polynomial:

    .. math::

        p(t) = \sum_{j=0}^{2} y_j \prod_{\substack{m=0 \\ m \neq j}}^{2}
               \frac{t - t_m}{t_j - t_m}

    The data in *y_data* may have arbitrary leading dimensions (e.g.
    ``(n, n_data)`` or ``(B, n, n_data)``); interpolation is always
    performed along the **last** axis.

    :param t_eval: query times of shape ``(n_eval,)``
    :type t_eval: torch.Tensor
    :param t_data: data times of shape ``(n_data,)``, must be sorted
    :type t_data: torch.Tensor
    :param y_data: data values with time along the last axis,
        shape ``(..., n_data)``
    :type y_data: torch.Tensor
    :returns: interpolated values, shape ``(..., n_eval)``
    :rtype: torch.Tensor
    """
    # Exact hit: querying at the data points themselves is the identity, so
    # skip the interpolation entirely.
    if torch.equal(t_eval, t_data):
        return y_data

    n_data = t_data.shape[0]

    # Find the index of the right neighbour for each query point
    # (searchsorted wants a contiguous boundary tensor; t_data is often a
    # strided slice, e.g. tsim[::save_every] from solve_ivp).
    idx = torch.searchsorted(t_data.contiguous(), t_eval).clamp(1, n_data - 1)

    # Choose the centre index of the 3-point stencil, clamped so that
    # i-1, i, i+1 are all valid
    ic = idx.clamp(1, n_data - 2)

    t0 = t_data[ic - 1]  # (n_eval,)
    t1 = t_data[ic]
    t2 = t_data[ic + 1]

    # Lagrange basis values at t_eval
    L0 = ((t_eval - t1) * (t_eval - t2)) / ((t0 - t1) * (t0 - t2))
    L1 = ((t_eval - t0) * (t_eval - t2)) / ((t1 - t0) * (t1 - t2))
    L2 = ((t_eval - t0) * (t_eval - t1)) / ((t2 - t0) * (t2 - t1))

    # Gather data values at the stencil points: (..., n_eval)
    y0 = y_data[..., ic - 1]
    y1 = y_data[..., ic]
    y2 = y_data[..., ic + 1]

    return y0 * L0 + y1 * L1 + y2 * L2
