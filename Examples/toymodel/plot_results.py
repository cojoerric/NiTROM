import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from NiTROM.Optimization_Functions import classes, utils
from NiTROM.PyTorch_Functions import gpu_utils, train, integrators
from NiTROM.PyTorch_Functions.linear_interpolation import Interp1D
import fom_class_pytorch
from plotting_utils import save_uncover_line_slides

# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams["legend.edgecolor"] = 'black'
# plt.rcParams["legend.fontsize"] = 14
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 16
# plt.rcParams['lines.linewidth'] = 2
torch.set_printoptions(precision=8)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.usetex": True,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "legend.handlelength": 2.8,
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

FIG_WIDTH = 3.4
FIG_WIDTH_WIDE = 6.8
FIG_HEIGHT = 2.6
TRAINING_SHADE = "#ececec"

cPOD, cOI, cOI_gs, cNIT, cNIT_gs = "#1b9e77", "#d95f02", "#a6761d", "#386cb0", "#7b3294"
lPOD, lOI, lOI_gs, lNIT, lNIT_gs = (
    (0, (1.0, 1.2)),
    (0, (5.0, 2.4)),
    "solid",
    (0, (8.0, 2.5)),
    "solid",
)


def make_figure(*, wide=False):
    width = FIG_WIDTH_WIDE if wide else FIG_WIDTH
    return plt.subplots(figsize=(width, FIG_HEIGHT))


def style_axes(ax, *, xlabel, ylabel, xlim=None, ylim=None, log_y=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        bottom, top = ylim
        ax.set_ylim(bottom=bottom, top=top)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", color="#d7d7d7", linewidth=0.55, alpha=0.7)
    ax.grid(which="minor", color="#efefef", linewidth=0.4, alpha=0.8)


def add_training_window(ax, x_end=10.0):
    ax.axvspan(0.0, x_end, color=TRAINING_SHADE, alpha=0.9, zorder=0)
    ax.text(
        0.16,
        0.88,
        "Training window",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            facecolor="white",
            edgecolor="#6f6f6f",
            linewidth=0.6,
            alpha=0.95,
            boxstyle="round,pad=0.25",
        ),
    )


def save_figure(fig, stem):
    fig.savefig(f"figures/{stem}.eps", format="eps")
    # fig.savefig(f"figures/{stem}.pdf", format="pdf")
    fig.savefig(f"figures/{stem}.png", format="png")
    plt.close(fig)


slides = False


device, rank, world_size = gpu_utils.setup_distributed_gpus()
dtype = torch.float64
if rank == 0:
    print(f"Using {world_size} devices for distributed training.")
    print(f"Device: {device}")

n = 3
n_traj = 4
beta = 20.0
diag_vec = torch.tensor([-1.0, -2.0, -5.0], device=device, dtype=dtype)
A2 = torch.diag(diag_vec)
A3 = torch.zeros((3, 3, 3), device=device, dtype=dtype)
diag_vec2 = torch.tensor([beta, beta, 0.0], device=device, dtype=dtype)
A3[:, :, -1] = torch.diag(diag_vec2)
B = torch.ones((3, 1), device=device, dtype=dtype)
C = torch.ones((1, 3), device=device, dtype=dtype)
fom = fom_class_pytorch.full_order_model(A2, A3, B, C, device=device, dtype=dtype)

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {
    "fname_steady_forcing": fname_forcing,
    "fname_weights": fname_weight,
    "fname_derivs": fname_deriv,
    "dtype": dtype,
    "device": device,
    "rank": rank,
    "world_size": world_size,
}
pool = classes.pool(*pool_inputs, **pool_kwargs)

r = 2  # ROM dimension
poly_comp = [1, 2]  # Model with a linear part and a quadratic part

# %% Compute NiTROM model

which_trajs = torch.arange(0, pool.n_traj, 1, device=device)
which_times = torch.arange(0, pool.n_snapshots, 1, device=device)
leggauss_deg = 5
nsave_rom = 2
integrator = integrators.my_rk4_adaptive

opt_obj_inputs = (pool, which_trajs, which_times, leggauss_deg, nsave_rom, poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

N = pool.n_snapshots * pool.n_traj
X = torch.zeros((pool.X.shape[1], N), device=device, dtype=dtype)
for i in range(pool.n_traj):
    X[:, i * pool.n_snapshots : (i + 1) * pool.n_snapshots] = pool.X[i,]
phi_pod, _, _ = torch.linalg.svd(X, full_matrices=False)
phi_pod = phi_pod[:, :r]
psi_pod = phi_pod.clone()

A_pod = torch.tensor(np.load("results/A_pod.npy"), device=device, dtype=dtype)
H_pod = torch.tensor(np.load("results/H_pod.npy"), device=device, dtype=dtype)
tensors_pod = (A_pod, H_pod)

A_oi = torch.tensor(np.load("results/A_oi.npy"), device=device, dtype=dtype)
H_oi = torch.tensor(np.load("results/H_oi.npy"), device=device, dtype=dtype)
tensors_oi = (A_oi, H_oi)

A_oi_gs = torch.tensor(np.load("results/A_oi_gs.npy"), device=device, dtype=dtype)
H_oi_gs = torch.tensor(np.load("results/H_oi_gs.npy"), device=device, dtype=dtype)
tensors_oi_gs = (A_oi_gs, H_oi_gs)

phi_nit = torch.tensor(np.load("results/phi_nit.npy"), device=device, dtype=dtype)
psi_nit = torch.tensor(np.load("results/psi_nit.npy"), device=device, dtype=dtype)
A_nit = torch.tensor(np.load("results/A_nit.npy"), device=device, dtype=dtype)
H_nit = torch.tensor(np.load("results/H_nit.npy"), device=device, dtype=dtype)
tensors_nit = (A_nit, H_nit)

phi_nit_gs = torch.tensor(np.load("results/phi_nit_gs.npy"), device=device, dtype=dtype)
psi_nit_gs = torch.tensor(np.load("results/psi_nit_gs.npy"), device=device, dtype=dtype)
A_nit_gs = torch.tensor(np.load("results/A_nit_gs.npy"), device=device, dtype=dtype)
H_nit_gs = torch.tensor(np.load("results/H_nit_gs.npy"), device=device, dtype=dtype)
tensors_nit_gs = (A_nit_gs, H_nit_gs)

# Plot errors
max_val = 5 / 20
betas = torch.rand(100, dtype=dtype, device=device) * 0.999 * max_val
t_eval = torch.linspace(0, 30, steps=300, device=device, dtype=dtype)
idx_10 = torch.argmin(torch.abs(t_eval - 10.0))
error_pod = torch.zeros_like(t_eval)
error_oi = torch.zeros_like(t_eval)
error_oi_gs = torch.zeros_like(t_eval)
error_nit = torch.zeros_like(t_eval)
error_nit_gs = torch.zeros_like(t_eval)

for k in range(len(betas)):
    u = betas[k] * torch.ones(n, device=device, dtype=dtype)
    x0 = torch.zeros(n, device=device, dtype=dtype)
    z0 = torch.zeros(r, device=device, dtype=dtype)

    sol = integrators.my_rk4_adaptive(fom.evaluate_fom_dynamics, t_eval, x0, args=(u,))
    id_ss = torch.tensor(
        [
            -betas[k] / (-1 + 4 * betas[k]),
            -betas[k] / (-2 + 4 * betas[k]),
            betas[k] / 5,
        ],
        device=device,
        dtype=dtype,
    )
    weight = torch.norm(fom.compute_output(id_ss)) ** 2

    sol_pod_r = integrators.my_rk4_adaptive(
        opt_obj.evaluate_rom_rhs, t_eval, z0, args=(psi_pod.T @ u,) + tensors_pod
    )
    sol_pod = phi_pod @ sol_pod_r
    error_pod += torch.norm(C @ (sol_pod - sol), dim=0) ** 2 / weight / len(betas)

    sol_oi_r = integrators.my_rk4_adaptive(
        opt_obj.evaluate_rom_rhs, t_eval, z0, args=(psi_pod.T @ u,) + tensors_oi
    )
    sol_oi = phi_pod @ sol_oi_r
    error_oi += torch.norm(C @ (sol_oi - sol), dim=0) ** 2 / weight / len(betas)

    sol_oi_gs_r = integrators.my_rk4_adaptive(
        opt_obj.evaluate_rom_rhs, t_eval, z0, args=(psi_pod.T @ u,) + tensors_oi_gs
    )
    sol_oi_gs = phi_pod @ sol_oi_gs_r
    error_oi_gs += torch.norm(C @ (sol_oi_gs - sol), dim=0) ** 2 / weight / len(betas)

    sol_nit_r = integrators.my_rk4_adaptive(
        opt_obj.evaluate_rom_rhs, t_eval, z0, args=(psi_nit.T @ u,) + tensors_nit
    )
    sol_nit = phi_nit @ sol_nit_r
    error_nit += torch.norm(C @ (sol_nit - sol), dim=0) ** 2 / weight / len(betas)

    sol_nit_gs_r = integrators.my_rk4_adaptive(
        opt_obj.evaluate_rom_rhs, t_eval, z0, args=(psi_nit_gs.T @ u,) + tensors_nit_gs
    )
    sol_nit_gs = phi_nit_gs @ sol_nit_gs_r
    error_nit_gs += torch.norm(C @ (sol_nit_gs - sol), dim=0) ** 2 / weight / len(betas)

fig, ax = make_figure()
ax.semilogy(
    t_eval.cpu()[:idx_10],
    error_pod.cpu()[:idx_10],
    label="POD-Gal.",
    color=cPOD,
    linestyle=lPOD,
)
ax.semilogy(
    t_eval.cpu()[:idx_10],
    error_oi.cpu()[:idx_10],
    label="OpInf",
    color=cOI,
    linestyle=lOI,
)
ax.semilogy(
    t_eval.cpu()[:idx_10],
    error_oi_gs.cpu()[:idx_10],
    label="GasOpInf",
    color=cOI_gs,
    linestyle=lOI_gs,
)
ax.semilogy(
    t_eval.cpu()[:idx_10],
    error_nit.cpu()[:idx_10],
    label="NiTROM",
    color=cNIT,
    linestyle=lNIT,
)
ax.semilogy(
    t_eval.cpu()[:idx_10],
    error_nit_gs.cpu()[:idx_10],
    label="GasNiTROM",
    color=cNIT_gs,
    linestyle=lNIT_gs,
)
style_axes(
    ax, xlabel="Time $t$", ylabel="Average error $e(t)$", xlim=(0.0, 10.0), log_y=True
)
save_figure(fig, "error_toymodel_10")

fig, ax = make_figure(wide=True)
ax.semilogy(t_eval.cpu(), error_pod.cpu(), label="POD-Gal.", color=cPOD, linestyle=lPOD)
ax.semilogy(t_eval.cpu(), error_oi.cpu(), label="OpInf", color=cOI, linestyle=lOI)
ax.semilogy(
    t_eval.cpu(), error_oi_gs.cpu(), label="GasOpInf", color=cOI_gs, linestyle=lOI_gs
)
ax.semilogy(t_eval.cpu(), error_nit.cpu(), label="NiTROM", color=cNIT, linestyle=lNIT)
ax.semilogy(
    t_eval.cpu(),
    error_nit_gs.cpu(),
    label="GasNiTROM",
    color=cNIT_gs,
    linestyle=lNIT_gs,
)
style_axes(
    ax,
    xlabel="Time $t$",
    ylabel="Average error $e(t)$",
    xlim=(0.0, 30.0),
    ylim=(None, 1e1),
    log_y=True,
)
add_training_window(ax)
ax.legend(loc="lower right", ncol=2, columnspacing=1.0, handletextpad=0.5)
save_figure(fig, "error_toymodel_30")

if slides:
    save_uncover_line_slides(
        t_eval.cpu(),
        [
            error_pod.cpu(),
            error_oi.cpu(),
            error_nit.cpu(),
            error_oi_gs.cpu(),
            error_nit_gs.cpu(),
        ],
        ["POD-Gal.", "OpInf", "NiTROM", "GasOpInf", "GasNiTROM"],
        "figures/slides/error_toymodel_uncover",
        colors=[cPOD, cOI, cNIT, cOI_gs, cNIT_gs],
        linestyles=[lPOD, lOI, lNIT, lOI_gs, lNIT_gs],
        linewidths=[2, 2, 2, 2, 2],
        xlabel="Time $t$",
        ylabel="Average error $e(t)$",
        xlim=(0.0, 30.0),
        ylim=(0.0, 1e1),
        legend_loc="lower right",
        fixed_legend_size=True,
        log_y=True,
    )


tk = torch.linspace(0, 30, steps=1000, device=device, dtype=dtype)
uk = 0.45 * (torch.sin(tk) + torch.cos(2 * tk))

t_eval = torch.linspace(0, 30, steps=1000, device=device, dtype=dtype)
idx_10 = torch.argmin(torch.abs(tk - 10.0))

fu = Interp1D(tk, torch.outer(B.flatten(), uk), extrapolate=True)
sol_fom = integrators.my_rk4_adaptive(
    fom.evaluate_fom_dynamics,
    tk,
    torch.zeros(3, device=device, dtype=dtype),
    args=(fu,),
)

fu = Interp1D(tk, torch.outer((psi_pod.T @ B).flatten(), uk), extrapolate=True)
sol_pod = phi_pod @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_pod,
)

fu = Interp1D(tk, torch.outer((psi_pod.T @ B).flatten(), uk), extrapolate=True)
sol_oi = phi_pod @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_oi,
)
sol_oi_gs = phi_pod @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_oi_gs,
)

fu = Interp1D(tk, torch.outer((psi_nit.T @ B).flatten(), uk), extrapolate=True)
sol_nit = phi_nit @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_nit,
)

fu = Interp1D(tk, torch.outer((psi_nit_gs.T @ B).flatten(), uk), extrapolate=True)
sol_nit_gs = phi_nit_gs @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_nit_gs,
)

fig, ax = make_figure()
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_fom)[0,][:idx_10].cpu(),
    color="k",
    linewidth=2.4,
    label="FOM",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_pod)[0,][:idx_10].cpu(),
    color=cPOD,
    linestyle=lPOD,
    label="POD-Gal.",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_nit)[0,][:idx_10].cpu(),
    color=cNIT,
    linestyle=lNIT,
    label="NiTROM",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_nit_gs)[0,][:idx_10].cpu(),
    color=cNIT_gs,
    linestyle=lNIT_gs,
    label="GasNiTROM",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_oi)[0,][:idx_10].cpu(),
    color=cOI,
    linestyle=lOI,
    label="OpInf",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_oi_gs)[0,][:idx_10].cpu(),
    color=cOI_gs,
    linestyle=lOI_gs,
    label="GasOpInf",
)
style_axes(ax, xlabel="Time $t$", ylabel="$y(t)$", xlim=(0.0, 10.0))
save_figure(fig, "response_toymodel_low_10")

fig, ax = make_figure(wide=True)
ax.plot(
    t_eval, fom.compute_output(sol_fom)[0,].cpu(), color="k", linewidth=2.4, label="FOM"
)
ax.plot(
    t_eval,
    fom.compute_output(sol_pod)[0,].cpu(),
    color=cPOD,
    linestyle=lPOD,
    label="POD-Gal.",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_oi)[0,].cpu(),
    color=cOI,
    linestyle=lOI,
    label="OpInf",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_nit)[0,].cpu(),
    color=cNIT,
    linestyle=lNIT,
    label="NiTROM",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_oi_gs)[0,].cpu(),
    color=cOI_gs,
    linestyle=lOI_gs,
    label="GasOpInf",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_nit_gs)[0,].cpu(),
    color=cNIT_gs,
    linestyle=lNIT_gs,
    label="GasNiTROM",
)
style_axes(ax, xlabel="Time $t$", ylabel="$y(t)$", xlim=(0.0, 30.0))
# add_training_window(ax)
save_figure(fig, "response_toymodel_low_30")

if slides:
    y_fom = fom.compute_output(sol_fom)[0,].cpu()
    y_pod = fom.compute_output(sol_pod)[0,].cpu()
    y_oi = fom.compute_output(sol_oi)[0,].cpu()
    y_nit = fom.compute_output(sol_nit)[0,].cpu()
    y_oi_gs = fom.compute_output(sol_oi_gs)[0,].cpu()
    y_nit_gs = fom.compute_output(sol_nit_gs)[0,].cpu()
    save_uncover_line_slides(
        t_eval,
        [y_fom, y_pod, y_oi, y_nit, y_oi_gs, y_nit_gs],
        ["FOM", "POD-Gal.", "OpInf", "NiTROM", "GasOpInf", "GasNiTROM"],
        "figures/slides/response_toymodel_low_uncover",
        colors=["k", cPOD, cOI, cNIT, cOI_gs, cNIT_gs],
        linestyles=["solid", lPOD, lOI, lNIT, lOI_gs, lNIT_gs],
        linewidths=[2, 2, 2, 2, 2, 2],
        xlabel="Time $t$",
        ylabel="$y(t)$",
        xlim=(0.0, 30.0),
        legend_loc="upper left",
        fixed_legend_size=True,
    )


tk = torch.linspace(0, 30, steps=1000, device=device, dtype=dtype)
idx_10 = torch.argmin(torch.abs(tk - 10.0))
uk = 0.65 * (torch.sin(tk) + torch.cos(2 * tk))

t_eval = torch.linspace(0, 30, steps=1000, device=device, dtype=dtype)

fu = Interp1D(tk, torch.outer(B.flatten(), uk), extrapolate=True)
sol_fom = integrators.my_rk4_adaptive(
    fom.evaluate_fom_dynamics,
    tk,
    torch.zeros(3, device=device, dtype=dtype),
    args=(fu,),
)

fu = Interp1D(tk, torch.outer((psi_pod.T @ B).flatten(), uk), extrapolate=True)
sol_pod = phi_pod @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_pod,
)

fu = Interp1D(tk, torch.outer((psi_pod.T @ B).flatten(), uk), extrapolate=True)
sol_oi = phi_pod @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_oi,
)
sol_oi_gs = phi_pod @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_oi_gs,
)

fu = Interp1D(tk, torch.outer((psi_nit.T @ B).flatten(), uk), extrapolate=True)
sol_nit = phi_nit @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_nit,
)

fu = Interp1D(tk, torch.outer((psi_nit_gs.T @ B).flatten(), uk), extrapolate=True)
sol_nit_gs = phi_nit_gs @ integrators.my_rk4_adaptive(
    opt_obj.evaluate_rom_rhs,
    tk,
    torch.zeros(r, device=device, dtype=dtype),
    args=(fu,) + tensors_nit_gs,
)

fig, ax = make_figure()
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_fom)[0,][:idx_10].cpu(),
    color="k",
    linewidth=2.4,
    label="FOM",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_pod)[0,][:idx_10].cpu(),
    color=cPOD,
    linestyle=lPOD,
    label="POD-Gal.",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_nit)[0,][:idx_10].cpu(),
    color=cNIT,
    linestyle=lNIT,
    label="NiTROM",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_nit_gs)[0,][:idx_10].cpu(),
    color=cNIT_gs,
    linestyle=lNIT_gs,
    label="GasNiTROM",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_oi)[0,][:idx_10].cpu(),
    color=cOI,
    linestyle=lOI,
    label="OpInf",
)
ax.plot(
    t_eval[:idx_10],
    fom.compute_output(sol_oi_gs)[0,][:idx_10].cpu(),
    color=cOI_gs,
    linestyle=lOI_gs,
    label="GasOpInf",
)
style_axes(
    ax,
    xlabel="Time $t$",
    ylabel="$y(t)$",
    xlim=(0.0, 10.0),
    ylim=(
        fom.compute_output(sol_fom)[:, :idx_10].min().item() * 2,
        fom.compute_output(sol_fom)[:, :idx_10].max().item() * 2.5,
    ),
)
save_figure(fig, "response_toymodel_high_10")

fig, ax = make_figure(wide=True)
ax.plot(
    t_eval, fom.compute_output(sol_fom)[0,].cpu(), color="k", linewidth=2.4, label="FOM"
)
ax.plot(
    t_eval,
    fom.compute_output(sol_pod)[0,].cpu(),
    color=cPOD,
    linestyle=lPOD,
    label="POD-Gal.",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_nit)[0,].cpu(),
    color=cNIT,
    linestyle=lNIT,
    label="NiTROM",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_nit_gs)[0,].cpu(),
    color=cNIT_gs,
    linestyle=lNIT_gs,
    label="GasNiTROM",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_oi)[0,].cpu(),
    color=cOI,
    linestyle=lOI,
    label="OpInf",
)
ax.plot(
    t_eval,
    fom.compute_output(sol_oi_gs)[0,].cpu(),
    color=cOI_gs,
    linestyle=lOI_gs,
    label="GasOpInf",
)
style_axes(
    ax,
    xlabel="Time $t$",
    ylabel="$y(t)$",
    xlim=(0.0, 30.0),
    ylim=(fom.compute_output(sol_fom).min().item() * 2, 12.0),
)
# add_training_window(ax)
save_figure(fig, "response_toymodel_high_30")

if slides:
    y_fom = fom.compute_output(sol_fom)[0,].cpu()
    y_pod = fom.compute_output(sol_pod)[0,].cpu()
    y_oi = fom.compute_output(sol_oi)[0,].cpu()
    y_nit = fom.compute_output(sol_nit)[0,].cpu()
    y_oi_gs = fom.compute_output(sol_oi_gs)[0,].cpu()
    y_nit_gs = fom.compute_output(sol_nit_gs)[0,].cpu()
    save_uncover_line_slides(
        t_eval,
        [y_fom, y_pod, y_oi, y_nit, y_oi_gs, y_nit_gs],
        ["FOM", "POD-Gal.", "OpInf", "NiTROM", "GasOpInf", "GasNiTROM"],
        "figures/slides/response_toymodel_high_uncover",
        colors=["k", cPOD, cOI, cNIT, cOI_gs, cNIT_gs],
        linestyles=["solid", lPOD, lOI, lNIT, lOI_gs, lNIT_gs],
        linewidths=[2, 2, 2, 2, 2, 2],
        xlabel="Time $t$",
        ylabel="$y(t)$",
        xlim=(0.0, 30.0),
        ylim=(-1, 12.0),
        legend_loc="upper left",
        fixed_legend_size=True,
        show_legend=False,
    )
