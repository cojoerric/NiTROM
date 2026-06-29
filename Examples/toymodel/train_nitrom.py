import os

import fom_class
import torch
import torch.distributed as dist

from nitrom.backend import cleanup_distributed, setup_distributed
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule, train
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

# Distributed setup (falls back to a single process when not launched with
# torchrun).  Run in parallel with, e.g.:
#     torchrun --standalone --nproc_per_node=2 train_nitrom.py
device, rank, world_size = setup_distributed()
dtype = torch.float64

traj_path = "./trajectories/"
models_dir = "./models/"
n_traj = 4
r = 2  # reduced dimension
poly_comp = [1, 2]

if rank == 0:
    os.makedirs(models_dir, exist_ok=True)


def printr(*args, **kwargs) -> None:
    """Print on rank 0 only."""
    if rank == 0:
        print(*args, **kwargs)


def global_cost(module) -> float:
    """Cost summed across ranks (``module()`` returns the local partial cost)."""
    cost = module().detach().clone()
    if world_size > 1:
        dist.all_reduce(cost, op=dist.ReduceOp.SUM)
    return cost.item()


def save_checkpoint(
    tensors, kind: str, path: str, Phi, Psi, gas_params=None
) -> None:
    """
    Save a self-contained NiTROM ROM checkpoint: the optimized trial basis Phi,
    test basis Psi, and the physical operators [A, H, B].
    """
    ckpt = {
        "kind": kind,
        "r": r,
        "poly_comp": poly_comp,
        "forcing_config": forcing_config,
        "Phi": Phi.detach().cpu().clone(),
        "Psi": Psi.detach().cpu().clone(),
        "tensors": [t.detach().cpu().clone() for t in tensors],
        "gas_params": gas_params,
    }
    torch.save(ckpt, path)
    print(f"saved -> {path}")


# %% Load the (rank-sharded) trajectories into a TrainingPool

pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy",
    dtype=dtype,
    device=device,
    rank=rank,
    world_size=world_size,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_forcing=traj_path + "forcing_%03d.pkl",
    fname_derivs=traj_path + "deriv_%03d.npy",
)

# %% POD basis (rank r).  compute_POD gathers on root and, by default,
# broadcasts the basis to every rank -- so Phi is identical everywhere.

U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (N, r)
projection = LinearProjection([Phi, Phi])  # orthogonal (Psi = Phi)

# B = encode(B_fom) (fixed, not learned); the toy FOM is driven by B_fom=ones(N,1).
B_fom = torch.ones(Phi.shape[0], 1, device=device, dtype=dtype)
B_r = projection.encode(B_fom.T).T  # fixed reduced input operator, (r, m)
forcing_config = {"forcing_exists": True, "B": B_r, "m": B_fom.shape[1]}

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=1,
)

# %% Setup the FOM
beta = 20.0
A2 = torch.diag(torch.tensor([-1.0, -2.0, -5.0], device=device, dtype=dtype))
A3 = torch.zeros((3, 3, 3), device=device, dtype=dtype)
A3[:, :, -1] = torch.diag(
    torch.tensor([beta, beta, 0.0], device=device, dtype=dtype)
)
B_op = torch.ones((3, 1), device=device, dtype=dtype)
C = torch.ones((1, 3), device=device, dtype=dtype)
fom = fom_class.full_order_model(A2, A3, B_op, C, device=device, dtype=dtype)

# Galerkin projection (Psi = Phi): (A_r, H_r), (B_r, C_r).
(A2r, A3r), (Br, _) = fom.assemble_petrov_galerkin_tensors(Phi, Phi)
if rank == 0:
    save_checkpoint(
        [A2r, A3r, Br], "galerkin", os.path.join(models_dir, "galerkin_model.pt"), Phi, Phi
    )

# %% 1) Train standard NiTROM
# printr("\n=== NiTROM ===")
# nitrom_model = PolynomialModel(
#     r, poly_comp, device=device, dtype=dtype, forcing_config=forcing_config, tensors=(A2r, A3r, Br)
# )
# registry = ParamRegistry(nitrom_model, projection)
# nitrom = NitromModule(training_data, registry, fom=fom, n_substeps=10)
# nitrom.set_unlearnable("B")  # B = Phi^T B_fom is fixed, not trained

# printr(f"initial cost: {global_cost(nitrom):.6e}")
# # We call train(). Since model is a NitromModule, Phi and Psi will automatically be treated
# # as Grassmann and Stiefel manifold elements respectively.
# train(
#     nitrom,
#     n_epochs=30,
#     lr=1.0,
#     optimizer_type="lbfgs",
#     print_every=1,
#     tol=1e-14,
# )
# printr(f"final cost:   {global_cost(nitrom):.6e}")

# if rank == 0:
#     nitrom._sync_to_registry()
#     save_checkpoint(
#         nitrom_model.get_params(),
#         "nitrom",
#         os.path.join(models_dir, "nitrom_model.pt"),
#         nitrom.projection.Phi,
#         nitrom.projection.Psi,
#     )
# nitrom_checkpoint = torch.load(os.path.join(models_dir, "nitrom_model.pt"))
# nitrom_model = PolynomialModel(
#     nitrom_checkpoint["r"], nitrom_checkpoint["poly_comp"], device=device, dtype=dtype, forcing_config=forcing_config, tensors=tuple(nitrom_checkpoint["tensors"])
# )
# registry = ParamRegistry(nitrom_model, projection)
# nitrom = NitromModule(training_data, registry, fom=fom, n_substeps=10)
# nitrom.set_unlearnable("B")

# %% 2) Train GAS-NiTROM, initialized from the Gas-OpInf operators
printr("\n=== GAS-NiTROM ===")
gas_opinf_checkpoint = torch.load(os.path.join(models_dir, "gas_opinf_model.pt"))
gas_init = [t.to(device=device, dtype=dtype) for t in gas_opinf_checkpoint["gas_params"]]

gas_nitrom_model = GasPolynomialModel(
    r,
    poly_comp,
    device=device,
    dtype=dtype,
    gas_params=gas_init,
    forcing_config=forcing_config,
)

# Start projection from the optimized Gas-OpInf bases
projection_gas = LinearProjection([Phi, Phi])

registry_gas = ParamRegistry(gas_nitrom_model, projection_gas)
gas_nitrom = NitromModule(training_data, registry_gas, fom=fom, n_substeps=10)
gas_nitrom.set_unlearnable("B")

printr(f"initial cost: {global_cost(gas_nitrom):.6e}")
train(
    gas_nitrom,
    n_epochs=30,
    lr=1.0,
    optimizer_type="lbfgs",
    print_every=1,
    tol=1e-14,
)
printr(f"final cost:   {global_cost(gas_nitrom):.6e}")

if rank == 0:
    gas_nitrom._sync_to_registry()
    gas_params = [
        t.detach().cpu().clone() for t in gas_nitrom_model.get_params()
    ]
    save_checkpoint(
        gas_nitrom_model.model.get_params(),
        "gas_nitrom",
        os.path.join(models_dir, "gas_nitrom_model.pt"),
        gas_nitrom.projection.Phi,
        gas_nitrom.projection.Psi,
        gas_params=gas_params,
    )

cleanup_distributed()
