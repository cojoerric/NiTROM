from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dill
import numpy as np

from .backend import (
    comm_rank_size,
    distributed_rank_size,
    get_backend,
    mpi_comm_world,
)


class TrainingPool:
    """Training data pool for trajectory-based optimization.

    Loads trajectory snapshots, optional per-trajectory weights, steady
    forcing fields, and time derivatives from disk, and distributes them
    across ranks when running in a multi-process setting.  Arrays are created
    with the active backend (NumPy or PyTorch).

    Parameters
    ----------
    n_traj : int
        Total number of trajectories to load from disk.
    fname_traj : str
        Format string for trajectory files (e.g., ``'traj_%03d.npy'``).
    fname_time : str
        Format string for the time file.
    dtype : optional
        Data type for all arrays. Defaults to the backend's ``float32``.
    device : str, optional
        Device on which arrays are allocated (ignored by NumPy). Default ``'cpu'``.
    comm : optional
        Communicator defining the process group over which trajectories are
        sharded -- either an ``mpi4py`` communicator (e.g. a split of
        ``COMM_WORLD``) or a ``torch.distributed`` process group; its rank/size
        are read through whichever API it exposes. If ``None`` (default), the
        multiprocessing context is auto-detected -- ``MPI.COMM_WORLD`` under
        ``mpiexec`` (NumPy backend), ``torch.distributed``/``torchrun`` for
        PyTorch, single-process otherwise -- so a parallel job shards correctly
        without the caller having to wire up rank/size by hand.
    **kwargs
        Optional ``fname_weights``, ``fname_forcing``, ``fname_derivs``.
    """

    def __init__(
        self,
        n_traj: int,
        fname_traj: str,
        fname_time: str,
        dtype: Any = None,
        device: str = "cpu",
        comm: Any = None,
        **kwargs: str,
    ) -> None:
        self.backend = get_backend()
        self.dtype = dtype if dtype is not None else self.backend.float32
        self.device = device
        # Figure out the process group ourselves: an explicit communicator wins;
        # otherwise auto-detect from the active backend (COMM_WORLD under
        # mpiexec, torch.distributed under torchrun, else single-process).  The
        # resolved communicator is stored so collectives (e.g. compute_POD) use
        # the same group the data was sharded over.
        if comm is not None:
            self.comm = comm
            self.rank, self.world_size = comm_rank_size(comm)
        else:
            self.rank, self.world_size = distributed_rank_size()
            # Keep COMM_WORLD around for numpy collectives (None if MPI absent).
            self.comm = mpi_comm_world() if self.backend.is_numpy else None

        if n_traj <= 0:
            raise ValueError(
                f"n_traj must be a positive integer. Currently, n_traj = {n_traj}."
            )
        self.n_traj = n_traj
        self.is_distributed = self.world_size > 1

        # Distribute trajectories across ranks
        self.my_n_traj = n_traj // self.world_size
        self.my_n_traj += 1 if self.rank < n_traj % self.world_size else 0

        if self.my_n_traj == 0:
            raise ValueError("Every rank needs to own at least one trajectory")

        start_idx = self.rank * (n_traj // self.world_size) + min(
            self.rank, n_traj % self.world_size
        )
        self.traj_indices = list(range(start_idx, start_idx + self.my_n_traj))

        # Load data from file
        self.load_trajectories(fname_traj)
        self.load_weights(kwargs)
        self.load_forcing(kwargs)
        self.load_time_derivatives(kwargs)
        self.time = self.backend.asarray(
            np.load(fname_time), dtype=self.dtype, device=self.device
        )

    def load_trajectories(self, fname_traj: str) -> None:
        """Load trajectory snapshots from ``.npy`` files into :attr:`X`."""
        self.fnames_traj = [fname_traj % k for k in self.traj_indices]
        X = [np.load(f) for f in self.fnames_traj]
        self.X = self.backend.asarray(
            np.stack(X), dtype=self.dtype, device=self.device
        )
        _, self.N, self.n_snapshots = self.X.shape

    def load_weights(self, kwargs: dict[str, str]) -> None:
        """Load per-trajectory importance weights (default: all ones)."""
        fname_weights: str | None = kwargs.get("fname_weights")
        if fname_weights is not None:
            self.fnames_weights = [fname_weights % k for k in self.traj_indices]
            weights = [np.load(f) for f in self.fnames_weights]
            self.weights = self.backend.asarray(
                np.stack(weights).reshape(-1), dtype=self.dtype, device=self.device
            )
        else:
            self.weights = self.backend.zeros(
                (self.my_n_traj,), dtype=self.dtype, device=self.device
            ) + 1.0

    def load_forcing(self, kwargs: dict[str, str]) -> None:
        """Load per-trajectory forcing callables from pickle files."""
        fname_forcing: str | None = kwargs.get("fname_forcing")
        if fname_forcing is not None:
            self.fnames_forcing = [fname_forcing % k for k in self.traj_indices]
            self.forcing_fns: list[Callable] = []
            for f in self.fnames_forcing:
                with open(f, "rb") as fh:
                    fn = dill.load(fh)
                if not callable(fn):
                    raise TypeError(f"Object loaded from {f} is not callable")
                self.forcing_fns.append(self._wrap_forcing(fn))
        else:
            self.forcing_fns = []

    def _wrap_forcing(self, fn: Callable) -> Callable:
        """Wrap a forcing callable so it returns a backend array of the right dtype."""

        def wrapped(t):
            return self.backend.asarray(
                fn(t), dtype=self.dtype, device=self.device
            )

        return wrapped

    def load_time_derivatives(self, kwargs: dict[str, str]) -> None:
        """Load precomputed time derivatives (default: zeros)."""
        fname_deriv: str | None = kwargs.get("fname_derivs")
        if fname_deriv is not None:
            self.fnames_deriv = [fname_deriv % k for k in self.traj_indices]
            dX = [np.load(f) for f in self.fnames_deriv]
            self.dX = self.backend.asarray(
                np.stack(dX), dtype=self.dtype, device=self.device
            )
        else:
            self.dX = self.backend.zeros(
                (self.my_n_traj, self.N, self.n_snapshots),
                dtype=self.dtype,
                device=self.device,
            )


class TrainingData:
    def __init__(
        self,
        pool,
        which_trajs,
        percent_time_length,
        leggauss_deg,
        nsave_rom,
        **kwargs,
    ):
        """
        Training-data view passed to the optimizer.

        pool:                an instance of TrainingPool
        which_trajs:         integer indices selecting a subset of pool's trajectories
        percent_time_length: fraction in (0, 1] of each trajectory's snapshots to use
        leggauss_deg:        number of Gauss-Legendre quadrature points for the gradient
        nsave_rom:           number of ROM snapshots stored between two FOM snapshots
        """
        self.pool = pool
        self.backend = pool.backend
        bkend = self.backend

        self.global_trajs = which_trajs
        self.local_trajs = self._global_to_local_indices(which_trajs)

        # Number of snapshots to keep from percent_time_length
        n_snapshots_total = pool.X.shape[2]
        n_keep = max(1, int(percent_time_length * n_snapshots_total))
        self.time = pool.time[:n_keep]

        if len(self.local_trajs) > 0:
            self.X = pool.X[self.local_trajs, :, :n_keep]
            self.dX = pool.dX[self.local_trajs, :, :n_keep]
            self.forcing_fns = [pool.forcing_fns[i] for i in self.local_trajs]
            self.weights = pool.weights[self.local_trajs]
        else:
            shape = (0, pool.N, n_keep)
            self.X = bkend.zeros(shape, device=pool.device, dtype=pool.dtype)
            self.dX = bkend.zeros(shape, device=pool.device, dtype=pool.dtype)
            self.forcing_fns = []
            self.weights = bkend.empty((0,), device=pool.device, dtype=pool.dtype)

        self.my_n_traj, _, self.n_snapshots = self.X.shape
        self.nsave_rom = nsave_rom

        # Gauss-Legendre quadrature points and weights
        self.leggauss_deg = leggauss_deg
        tlg, wlg = np.polynomial.legendre.leggauss(self.leggauss_deg)
        self.tlg = bkend.asarray(tlg, device=pool.device, dtype=pool.dtype)
        self.wlg = bkend.asarray(wlg, device=pool.device, dtype=pool.dtype)

        # Scale the weights so the cost measures the average error over
        # snapshots and trajectories.
        self.weights = self.weights * (len(self.global_trajs) * self.n_snapshots)

        # Parse the keyword arguments
        self.which_fix = kwargs.get("which_fix", "fix_none")
        if self.which_fix not in ["fix_tensors", "fix_bases", "fix_none"]:
            raise ValueError("which_fix must be fix_none, fix_tensors or fix_bases")

        self.l2_pen = kwargs.get("stab_promoting_pen")
        self.pen_tf = kwargs.get("stab_promoting_tf")
        self.randic = kwargs.get("stab_promoting_ic")

        if self.l2_pen is not None and self.pen_tf is None:
            raise ValueError(
                "If you provide a value for stab_promoting_pen you also have "
                "to provide a value for stab_promoting_tf"
            )

        if self.l2_pen is not None and self.randic is None:
            raise ValueError(
                "If you provide a value for stab_promoting_pen you also have "
                "to provide a random ic vector of the same size as the ROM"
            )

        if self.randic is not None:
            self.randic = self.randic / bkend.vector_norm(self.randic)
            self.randic = self.randic.reshape(-1)

    def _global_to_local_indices(self, global_indices):
        bkend = self.backend
        global_indices = bkend.asarray(global_indices)
        gpu_indices = bkend.asarray(self.pool.traj_indices)

        # Filter to trajectories owned by this pool, then map global IDs to
        # local positions.
        mask = bkend.isin(global_indices, gpu_indices)
        requested_and_owned = global_indices[mask]

        sort_order = bkend.argsort(gpu_indices)
        sorted_gpu = gpu_indices[sort_order]
        positions = bkend.searchsorted(sorted_gpu, requested_and_owned)
        return sort_order[positions]
