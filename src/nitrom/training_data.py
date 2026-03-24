from __future__ import annotations

import dill
from collections.abc import Callable

import numpy as np
import torch


class TrainingPool:
    """Training data pool for trajectory-based optimization.

    Loads trajectory snapshots, optional per-trajectory weights, steady
    forcing fields, and time derivatives from disk, and distributes them
    across ranks when running in a multi-GPU setting.

    Parameters
    ----------
    n_traj : int
        Total number of trajectories to load from disk.
    fname_traj : str
        Format string for trajectory files (e.g., ``'traj_%03d.npy'``).
    fname_time : str
        Format string for per-trajectory time files
        (e.g., ``'time_%03d.npy'``).
    dtype : torch.dtype, optional
        Data type for all tensors. Default is ``torch.float32``.
    device : str | torch.device, optional
        Device on which tensors are allocated. Default is ``'cpu'``.
    rank : int, optional
        Rank of the current process in distributed training. Default is 0.
    world_size : int, optional
        Total number of processes. Default is 1 (single-process).
    **kwargs
        Optional keyword arguments:

        - **fname_weights** (*str*) -- Format string for per-trajectory
          weight files (e.g., ``'weight_%03d.npy'``).
        - **fname_forcing** (*str*) -- Format string for forcing
          callable files (e.g., ``'forcing_%03d.pkl'``).  Each file
          must contain a pickled callable with signature ``f(t) -> array``.
        - **fname_derivs** (*str*) -- Format string for time-derivative
          files (e.g., ``'fname_derivs_%03d.npy'``).

    Attributes
    ----------
    X : torch.Tensor
        Trajectory data with shape ``(my_n_traj, N, n_snapshots)``.
    times : list[torch.Tensor]
        Per-trajectory time vectors, each with shape ``(n_snapshots,)``.
    weights : torch.Tensor
        Per-trajectory weights with shape ``(my_n_traj,)``.
    forcing_fns : list[Callable[[torch.Tensor], torch.Tensor]]
        Per-trajectory forcing callables loaded from pickle files.
    dX : torch.Tensor
        Time derivatives with shape ``(my_n_traj, N, n_snapshots)``.
    N : int
        Spatial dimension of each trajectory.
    n_snapshots : int
        Number of time snapshots per trajectory.
    my_n_traj : int
        Number of trajectories assigned to this rank.
    """

    def __init__(
        self,
        n_traj: int,
        fname_traj: str,
        fname_time: str,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        rank: int = 0,
        world_size: int = 1,
        **kwargs: str,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.world_size = world_size

        if n_traj <= 0:
            raise ValueError(
                f"n_traj must be a positive integer. Currently, n_traj = {n_traj}."
            )
        self.n_traj = n_traj
        self.is_distributed = self.world_size > 1

        # Distribute trajectories across GPUs
        self.my_n_traj = n_traj // self.world_size
        self.my_n_traj += 1 if self.rank < n_traj % self.world_size else 0

        if self.my_n_traj == 0:
            raise ValueError(f"Every GPU needs to own at least one trajectory")

        start_idx = self.rank * (n_traj // self.world_size) + min(
            self.rank, n_traj % self.world_size
        )
        self.traj_indices = list(range(start_idx, start_idx + self.my_n_traj))

        # Load data from file
        self.load_trajectories(fname_traj)
        self.load_weights(kwargs)
        self.load_forcing(kwargs)
        self.load_time_derivatives(kwargs)
        self.time = torch.tensor(np.load(fname_time), device=self.device, dtype=self.dtype)

    def load_trajectories(self, fname_traj: str) -> None:
        """Load trajectory snapshots from ``.npy`` files.

        Populates :attr:`X` with shape ``(my_n_traj, N, n_snapshots)`` and
        sets :attr:`N` and :attr:`n_snapshots`.  Every rank is guaranteed to
        own at least one trajectory (enforced in ``__init__``).

        Parameters
        ----------
        fname_traj : str
            Format string that accepts a trajectory index
            (e.g., ``'traj_%03d.npy'``).
        """
        self.fnames_traj = [fname_traj % k for k in self.traj_indices]
        X = [np.load(f) for f in self.fnames_traj]
        self.X = torch.tensor(np.stack(X), device=self.device, dtype=self.dtype)
        _, self.N, self.n_snapshots = self.X.shape

    def load_weights(self, kwargs: dict[str, str]) -> None:
        """Load per-trajectory importance weights.

        If ``fname_weights`` is provided in *kwargs*, weights are read from
        the corresponding ``.npy`` files; otherwise every trajectory receives
        a weight of 1.

        Parameters
        ----------
        kwargs : dict[str, str]
            Must originate from the constructor keyword arguments.  The
            recognised key is ``fname_weights``.
        """
        fname_weights: str | None = kwargs.get("fname_weights", None)
        if fname_weights is not None:
            self.fnames_weights = [fname_weights % k for k in self.traj_indices]
            weights = [np.load(f) for f in self.fnames_weights]
            self.weights = torch.tensor(
                np.stack(weights), device=self.device, dtype=self.dtype
            )
        else:
            self.weights = torch.ones(
                self.my_n_traj, device=self.device, dtype=self.dtype
            )

    def load_forcing(self, kwargs: dict[str, str]) -> None:
        """Load per-trajectory forcing callables from pickle files.

        Each file must contain a pickled callable with signature
        ``f(t) -> array_like``.  The callable is validated by calling it
        with ``self.time``; the result is converted to a :class:`torch.Tensor`
        on the correct device and dtype if needed.

        If ``fname_forcing`` is not provided, :attr:`forcing_fns` is set to
        an empty list.

        Parameters
        ----------
        kwargs : dict[str, str]
            Must originate from the constructor keyword arguments.  The
            recognised key is ``fname_forcing``.
        """
        fname_forcing: str | None = kwargs.get("fname_forcing", None)
        if fname_forcing is not None:
            self.fnames_forcing = [fname_forcing % k for k in self.traj_indices]
            self.forcing_fns: list[Callable[[torch.Tensor], torch.Tensor]] = []
            for f in self.fnames_forcing:
                with open(f, "rb") as fh:
                    fn = dill.load(fh)
                if not callable(fn):
                    raise TypeError(f"Object loaded from {f} is not callable")
                self.forcing_fns.append(self._wrap_forcing(fn))
        else:
            self.forcing_fns = []

    def _wrap_forcing(
        self, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Wrap a forcing callable so it always returns the correct type/device/dtype."""

        def wrapped(t: torch.Tensor) -> torch.Tensor:
            result = fn(t)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, device=self.device, dtype=self.dtype)
            if result.device != torch.device(self.device):
                result = result.to(device=self.device)
            if result.dtype != self.dtype:
                result = result.to(dtype=self.dtype)
            return result

        return wrapped

    def load_time_derivatives(self, kwargs: dict[str, str]) -> None:
        """Load precomputed time derivatives of the trajectories.

        If ``fname_derivs`` is provided in *kwargs*, derivatives are read
        from ``.npy`` files; otherwise :attr:`dX` is filled with zeros.

        Parameters
        ----------
        kwargs : dict[str, str]
            Must originate from the constructor keyword arguments.  The
            recognised key is ``fname_derivs``.
        """
        fname_deriv: str | None = kwargs.get("fname_derivs", None)
        if fname_deriv is not None:
            self.fnames_deriv = [fname_deriv % k for k in self.traj_indices]
            dX = [np.load(f) for f in self.fnames_deriv]
            self.dX = torch.tensor(np.stack(dX), device=self.device, dtype=self.dtype)
        else:
            self.dX = torch.zeros(
                (self.my_n_traj, self.N, self.n_snapshots),
                device=self.device,
                dtype=self.dtype,
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
        This class contains the training data information that will get passed to the optimizer.

        pool:           an instance of the pool class
        which_trajs:    array of integers to extract a subset of the trajectories contained in
                        pool.X. Useful if we end up using stochastic gradient descent
        percent_time_length: float in (0, 1] specifying the fraction of each trajectory's snapshots
                        to use. E.g., 0.1 keeps the first 10%. Useful for curriculum training where
                        we start on short trajectories and progressively extend them
        leggauss_deg:   number of Gauss-Legendre quadrature points used to approximate the integrals
                        in the gradient (see Prop. 2.1 in NiTROM arXiv paper)
        nsave_rom:      number of ROM snapshots to store in between two adjacent FOM snapshots
        Optional keyword arguments:
            which_fix:              one of fix_bases, fix_tensors or fix_none (default is fix_none)
            stab_promoting_pen:     value of L2 regularization coefficient
            stab_promoting_tf:      value of final time for stability promoting penalty
            stab_promoting_ic:      random (unit-norm) vector to probe the stability penalty
        """

        self.pool = pool

        self.global_trajs = which_trajs
        self.local_trajs = self._global_to_local_indices(which_trajs)

        # Compute number of snapshots to keep from percent_time_length
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
            self.X = torch.zeros(shape, device=pool.device, dtype=pool.dtype)
            self.dX = torch.zeros(shape, device=pool.device, dtype=pool.dtype)
            self.forcing_fns = []
            self.weights = torch.empty((0,), device=pool.device, dtype=pool.dtype)
        
        self.my_n_traj, _, self.n_snapshots = self.X.shape
        self.nsave_rom = nsave_rom

        # Gauss-Legendre quadrature points and weights
        # Cubic spline interpolation to compute integral
        self.leggauss_deg = leggauss_deg
        tlg, wlg = np.polynomial.legendre.leggauss(self.leggauss_deg)
        self.tlg = torch.tensor(tlg, device=pool.device, dtype=pool.dtype)
        self.wlg = torch.tensor(wlg, device=pool.device, dtype=pool.dtype)

        # Scale the weight accordingly so that the cost function measures
        # the average error over snapshots and trajectories.
        self.weights *= len(self.global_trajs) * self.n_snapshots

        # Parse the keyword arguments
        self.which_fix = kwargs.get("which_fix", "fix_none")
        if self.which_fix not in ["fix_tensors", "fix_bases", "fix_none"]:
            raise ValueError("which_fix must be fix_none, fix_tensors or fix_bases")

        self.l2_pen = kwargs.get("stab_promoting_pen", None)
        self.pen_tf = kwargs.get("stab_promoting_tf", None)
        self.randic = kwargs.get("stab_promoting_ic", None)

        if self.l2_pen != None and self.pen_tf == None:
            raise ValueError(
                "If you provide a value for stab_promoting_pen you \
                              also have to provide a value for stab_promoting_tf"
            )

        if self.l2_pen != None and self.randic == None:
            raise ValueError(
                "If you provide a value for stab_promoting_pen you \
                              also have to provide a random ic vector of the same \
                              size as the ROM"
            )

        if self.randic != None:
            self.randic /= torch.linalg.vector_norm(self.randic)
            self.randic = self.randic.reshape(-1)

    def _global_to_local_indices(self, global_indices):

        dev = self.pool.device
        # Convert global_indices to torch if necessary
        if not isinstance(global_indices, torch.Tensor):
            global_indices = torch.tensor(
                global_indices, device=self.pool.device, dtype=torch.long
            )

        # Filter to trajectories owned by this pool, then map global IDs to local positions
        gpu_indices = torch.tensor(self.pool.traj_indices, device=dev)
        mask = torch.isin(global_indices, gpu_indices)
        requested_and_owned = global_indices[mask]

        sorted_gpu, sort_order = gpu_indices.sort()
        positions = torch.searchsorted(sorted_gpu, requested_and_owned)
        local_indices = sort_order[positions]

        return local_indices