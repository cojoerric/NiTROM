import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

import time_steppers as tstep
import post_process as pp
from plotting_utils import save_uncover_line_slides, save_uncover_multiaxes_slides

from NiTROM.Optimization_Functions import classes
from NiTROM.PyTorch_Functions import gpu_utils, linear_interpolation
from NiTROM.PyTorch_Functions.integrators import my_rk4_adaptive
import classes_cavity

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
        # "xtick.top": True,
        # "ytick.right": True,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "legend.handlelength": 2.8,
        "figure.dpi": 140,
        "savefig.dpi": 300,
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
lPOD, lOI, lOI_gs, lNIT, lNIT_gs = (0, (1.0, 1.2)), (0, (5.0, 2.4)), "solid", (0, (8.0, 2.5)), "solid"


def make_figure(*, wide=False, nrows=1, ncols=1, height=None):
    width = FIG_WIDTH_WIDE if wide else FIG_WIDTH
    fig_height = height if height is not None else FIG_HEIGHT
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, fig_height), constrained_layout=True)


def style_axes(ax, *, xlabel=None, ylabel=None, xlim=None, ylim=None, log_y=False):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
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


def add_training_window(ax, *, x_end=20.0, text_xy=(0.26, 0.12)):
    ax.axvspan(0.0, x_end, color=TRAINING_SHADE, alpha=0.9, zorder=0)
    ax.text(
        text_xy[0],
        text_xy[1],
        "Training window",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#6f6f6f", linewidth=0.6, alpha=0.95, boxstyle="round,pad=0.25"),
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
    print(f"Using {world_size} GPU(s) for distributed training.")
    print(f"Device: {device}")

##
Lx = 1
Ly = 1
Nx = 100
Ny = 100

dx = Lx/Nx
dy = Ly/Ny
Re = 8300

flow = classes_cavity.flow_class(Lx,Ly,Nx,Ny,Re)

n = 400
dt = 1.0/n

lops = classes_cavity.linear_operators_2D(flow,dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
fom = classes_cavity.fom_class(flow,lops)
fom.assemble_forcing_profile(0.95, 0.05)
B = fom.f.copy()
B_torch = torch.tensor(B, device=device, dtype=dtype)

##
traj_path = "./trajectories/"

which = 'train'
if which == 'train':
    fname_traj = traj_path + "traj_%03d.npy"
    fname_weight = traj_path + "weight_%03d.npy"
    fname_deriv = traj_path + "deriv_%03d.npy"
    fname_time = traj_path + "time.npy"
    amps = np.load(traj_path + "amps.npy")
else:
    fname_traj = traj_path + "traj_%03d_testing.npy"
    fname_weight = traj_path + "weight_%03d_testing.npy"
    fname_deriv = traj_path + "deriv_%03d_testing.npy"
    fname_time = traj_path + "time.npy"
    amps = np.load(traj_path + "amps_testing.npy")

phi_pre = np.load(traj_path + "phi_pre.npy")
phi_pre_torch = torch.tensor(phi_pre, device=device, dtype=dtype)
n_traj = len(amps)
n = phi_pre.shape[-1]

pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
               'dtype':dtype,
               'device':device,
               'rank':rank,
               'world_size':world_size
}
pool = classes.pool(*pool_inputs,**pool_kwargs)

r = 50               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

which_trajs = torch.arange(0,pool.n_traj,1,device=device)
which_times = torch.arange(0,pool.n_snapshots,1,device=device)
leggauss_deg = 5
nsave_rom = 15

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

phi_pod = torch.eye(n, r, device=device, dtype=dtype)
psi_pod = phi_pod.clone()
A_pod = torch.tensor(np.load('results/A_pod.npy'), device=device, dtype=dtype)
H_pod = torch.tensor(np.load('results/H_pod.npy'), device=device, dtype=dtype)
tensors_pod = (A_pod, H_pod)

A_oi = torch.tensor(np.load('results/A_oi.npy'), device=device, dtype=dtype)
H_oi = torch.tensor(np.load('results/H_oi.npy'), device=device, dtype=dtype).reshape(r,r,r)
tensors_oi = (A_oi, H_oi)

A_oi_gs = torch.tensor(np.load('results/A_oi_gs.npy'), device=device, dtype=dtype)
H_oi_gs = torch.tensor(np.load('results/H_oi_gs.npy'), device=device, dtype=dtype)
tensors_oi_gs = (A_oi_gs, H_oi_gs)

phi_nit = torch.tensor(np.load('results/phi_nit.npy'), device=device, dtype=dtype)
psi_nit = torch.tensor(np.load('results/psi_nit.npy'), device=device, dtype=dtype)
A_nit = torch.tensor(np.load('results/A_nit.npy'), device=device, dtype=dtype)
H_nit = torch.tensor(np.load('results/H_nit.npy'), device=device, dtype=dtype).reshape(r,r,r)
tensors_nit = (A_nit, H_nit)

phi_nit_gs = torch.tensor(np.load('results/phi_nit_gs.npy'), device=device, dtype=dtype)
psi_nit_gs = torch.tensor(np.load('results/psi_nit_gs.npy'), device=device, dtype=dtype)
A_nit_gs = torch.tensor(np.load('results/A_nit_gs.npy'), device=device, dtype=dtype)
H_nit_gs = torch.tensor(np.load('results/H_nit_gs.npy'), device=device, dtype=dtype)
tensors_nit_gs = (A_nit_gs, H_nit_gs)


## Plot results
# Energy of perturbations
if which == 'train':
    fig, ax = make_figure(wide=False)
    for k in range (pool.my_n_traj):
        Qk = pool.X[k,]
        energy_k = torch.linalg.vector_norm(Qk,dim=0)**2
        ax.plot(pool.time.cpu().numpy(), energy_k.cpu().numpy(), color='k', alpha=0.85)
    style_axes(ax, xlabel=r'Time $t$', ylabel='Energy of perturbations', xlim=(0, pool.time[-1].item()))
    ax.set_ylim(bottom=0)
    add_training_window(ax, text_xy=(0.25, 0.78))
    save_figure(fig, "cavity_energy_perturbations")


# Testing/training error
time = pool.time.cpu().numpy()
u = torch.zeros(r, device=device, dtype=dtype)
e_pod = 0
e_oi = 0
e_oi_gs = 0
e_nit = 0
e_nit_gs = 0

idx_final = 80
fig, ax = make_figure(wide=False)
for k in range(n_traj):
    mean_en = torch.mean(torch.linalg.norm(pool.X[k,], dim=0)**2)

    # POD
    z_pod = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_pod, args=(u,) + tensors_pod)
    e_pod += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # if k == 0:
    #     plt.plot(time, e_pod.cpu().numpy(), color=cPOD, linestyle=lPOD, alpha=1.0, label='POD-Gal.')
    # else:
    #     plt.plot(time, e_pod.cpu().numpy(), color=cPOD, linestyle=lPOD, alpha=0.3)

    # OpInf
    z_oi = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_oi, args=(u,) + tensors_oi)
    e_oi += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # if k == 0:
    #     plt.plot(time, e_oi.cpu().numpy(), color=cOI, linestyle=lOI, alpha=1.0, label='OpInf')
    # else:
    #     plt.plot(time, e_oi.cpu().numpy(), color=cOI, linestyle=lOI, alpha=0.3)

    # OpInf GS
    z_oi_gs = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_oi_gs, args=(u,) + tensors_oi_gs)
    e_oi_gs += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # if k == 0:
    #     plt.plot(time, e_oi_gs.cpu().numpy(), color=cOI_gs, linestyle=lOI_gs, alpha=1.0, label='GsOpInf')
    # else:
    #     plt.plot(time, e_oi_gs.cpu().numpy(), color=cOI_gs, linestyle=lOI_gs, alpha=0.3)

    # NiTROM
    z_nit = psi_nit.T @ pool.X[k,:,0]
    sol = phi_nit @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_nit, args=(u,) + tensors_nit)
    e_nit += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # if k == 0:
    #     plt.plot(time, e_nit.cpu().numpy(), color=cNIT, linestyle=lNIT, alpha=1.0, label='NiTROM')
    # else:
    #     plt.plot(time, e_nit.cpu().numpy(), color=cNIT, linestyle=lNIT, alpha=0.3)

    # NiTROM GS
    z_nit_gs = psi_nit_gs.T @ pool.X[k,:,0]
    sol = phi_nit_gs @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_nit_gs, args=(u,) + tensors_nit_gs)
    e_nit_gs += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # if k == 0:
    #     plt.plot(time, e_nit_gs.cpu().numpy(), color=cNIT_gs, linestyle=lNIT_gs, alpha=1.0, label='GsNiTROM')
    # else:
    #     plt.plot(time, e_nit_gs.cpu().numpy(), color=cNIT_gs, linestyle=lNIT_gs, alpha=0.3)

ax.plot(time, e_pod.cpu().numpy(), color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(time, e_oi.cpu().numpy(), color=cOI, linestyle=lOI, label='OpInf')
ax.plot(time, e_oi_gs.cpu().numpy(), color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
ax.plot(time, e_nit.cpu().numpy(), color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(time, e_nit_gs.cpu().numpy(), color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
style_axes(ax, xlabel='Time $t$', ylabel='Error', xlim=(0, pool.time[idx_final].item()), ylim=(1e-3, None), log_y=True)
save_figure(fig, f'cavity_50_error_{which}_trained')

fig, ax = make_figure(wide=True)
ax.plot(time, e_pod.cpu().numpy(), color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(time, e_oi.cpu().numpy(), color=cOI, linestyle=lOI, label='OpInf')
ax.plot(time, e_oi_gs.cpu().numpy(), color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
ax.plot(time, e_nit.cpu().numpy(), color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(time, e_nit_gs.cpu().numpy(), color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
ax.set_xlim(left=0, right=pool.time[-1])
style_axes(ax, xlabel='Time $t$', ylabel='Error', xlim=(0, pool.time[-1].item()), ylim=(1e-3, None), log_y=True)
add_training_window(ax)
ax.legend(loc='upper right', ncol=3, columnspacing=1.0, handletextpad=0.5)
save_figure(fig, f'cavity_50_error_{which}_full')

if slides:
    save_uncover_line_slides(
        time,
        [e_pod.cpu().numpy(), e_oi.cpu().numpy(), e_oi_gs.cpu().numpy(), e_nit.cpu().numpy(), e_nit_gs.cpu().numpy()],
        ['POD-Gal.', 'OpInf', 'GasOpInf', 'NiTROM', 'GasNiTROM'],
        'figures/slides/cavity_error_test_uncover',
        colors=[cPOD, cOI, cOI_gs, cNIT, cNIT_gs],
        linestyles=[lPOD, lOI, lOI_gs, lNIT, lNIT_gs],
        linewidths=[2, 2, 2, 2, 2],
        xlabel='Time $t$',
        ylabel='Average error $e(t)$',
        xlim=(0.0, pool.time[-1].item()),
        ylim=(1e-3, None),
        legend_loc='upper right',
        fixed_legend_size=True,
        log_y=True,
        window_shading=True,
        show_legend=False,
    )
    


# Sinusoidal forcing
time_np = dt*np.arange(0, 80*n, 1)
time_torch = torch.tensor(time_np, device=device, dtype=dtype)
nsave = 5
amp = 0.1
energies = []
ks = [1,2,4]

for (k, harmonic) in enumerate(ks):
    freq = 1.00 * harmonic
    tf = np.arange(0, 2*torch.pi/freq, dt)
    fint = sp.interpolate.interp1d(tf, amp * np.sin(freq * tf), kind='linear', fill_value="extrapolate")
    print("Simulating trajectory with forcing frequency %.2f..." % freq)

    qic = flow.q_sbf.copy()
    dataf, tsavef = tstep.solver_2D(flow,lops,qic,time_np,nsave,[0,1,1,0,0,0,0,0],[1],[fint],[2*np.pi/freq],vol_forcing=B)
    dataf_torch = torch.tensor(dataf, device=device, dtype=dtype)
    qic_torch = torch.tensor(qic, device=device, dtype=dtype)
    energy_true = torch.linalg.vector_norm(dataf_torch - qic_torch.reshape(-1,1), dim=0)**2

    z0 = torch.zeros(r, device=device, dtype=dtype)

    # POD
    fpod = torch.einsum('i,j', psi_pod.T @ phi_pre_torch.T @ B_torch, amp*torch.sin(freq*time_torch))
    fpod = linear_interpolation.Interp1D(time_torch, fpod, extrapolate=True)
    sol_pod = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, time_torch[::nsave], z0, args=(fpod,) + tensors_pod)
    energy_pod = torch.linalg.vector_norm(sol_pod, dim=0)**2

    # OpInf
    foi = torch.einsum('i,j', psi_pod.T @ phi_pre_torch.T @ B_torch, amp*torch.sin(freq*time_torch))
    foi = linear_interpolation.Interp1D(time_torch, foi, extrapolate=True)
    sol_oi = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, time_torch[::nsave], z0, args=(foi,) + tensors_oi)
    energy_oi = torch.linalg.vector_norm(sol_oi, dim=0)**2

    # GsOpInf
    foi_gs = torch.einsum('i,j', psi_pod.T @ phi_pre_torch.T @ B_torch, amp*torch.sin(freq*time_torch))
    foi_gs = linear_interpolation.Interp1D(time_torch, foi_gs, extrapolate=True)
    sol_oi_gs = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, time_torch[::nsave], z0, args=(foi_gs,) + tensors_oi_gs)
    energy_oi_gs = torch.linalg.vector_norm(sol_oi_gs, dim=0)**2

    # NiTROM
    fnit = torch.einsum('i,j', psi_nit.T @ phi_pre_torch.T @ B_torch, amp*torch.sin(freq*time_torch))
    fnit = linear_interpolation.Interp1D(time_torch, fnit, extrapolate=True)
    sol_nit = phi_nit @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, time_torch[::nsave], z0, args=(fnit,) + tensors_nit)
    energy_nit = torch.linalg.vector_norm(sol_nit, dim=0)**2

    # GsNiTROM
    fnit_gs = torch.einsum('i,j', psi_nit_gs.T @ phi_pre_torch.T @ B_torch, amp*torch.sin(freq*time_torch))
    fnit_gs = linear_interpolation.Interp1D(time_torch, fnit_gs, extrapolate=True)
    sol_nit_gs = phi_nit_gs @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, time_torch[::nsave], z0, args=(fnit_gs,) + tensors_nit_gs)
    energy_nit_gs = torch.linalg.vector_norm(sol_nit_gs, dim=0)**2

    energy_lst = [energy_true, energy_pod, energy_oi, energy_oi_gs, energy_nit, energy_nit_gs]
    energies.append(energy_lst)

colors = ['k', cPOD, cOI, cOI_gs, cNIT, cNIT_gs]
lstyle = ['-', lPOD, lOI, lOI_gs, lNIT, lNIT_gs]
fig, ax = plt.subplots(nrows=3, ncols=1)
for k in range(len(energies)):
    for (i, vec) in enumerate(energies[k]):
        ax[k].plot(tsavef, vec.cpu().numpy(), color=colors[i], linestyle=lstyle[i])
        if k < len(energies) - 1:
            ax[k].set_xticklabels([])
    ax[k].text(
        0.03, 0.65, rf'$k = {ks[k]}$',
        transform=ax[k].transAxes,
        ha='left', va='bottom',
        fontsize=22,
    )
    ax[k].set_xlim(0, tsavef[-1])
    if k == 0:
        ax[k].set_ylim(bottom=0)
    elif k == 1:
        ax[k].set_ylim(0, energies[k][-2].cpu().numpy().max()*1.1)
    else:
        ax[k].set_ylim(0, energies[k][-1].cpu().numpy().max()*1.1)
    # ax[k].set_ylim(bottom=0)
    # style_axes(ax[k], xlim=(0, tsavef[-1]), ylim=(0, None))
    style_axes(ax[k], xlim=(0, tsavef[-1]), ylim=(0, ax[k].get_ylim()[1]))
    ax[k].grid(which="minor", visible=False)

ax[-1].set_xlabel(r'Time $t$')
# ax[-1].tick_params(top=False, bottom=True)
ax[1].set_ylabel('Energy')
fig.set_size_inches(FIG_WIDTH_WIDE, 6.6)
save_figure(fig, "cavity_50_forcing_0p1_energy")

if slides:
    y_groups = [[vec.cpu().numpy() for vec in energy_group] for energy_group in energies]
    # ylims = [
    #     (0.0, None),
    #     (0.0, energies[1][-2].cpu().numpy().max() * 1.1),
    #     (0.0, energies[2][-1].cpu().numpy().max() * 1.1),
    # ]
    ylims = [(0.0, None), (0.0, None), (0.0, None)]
    texts = [
        {'x': 0.03, 'y': 0.65, 'text': rf'$k = {ks[0]}$', 'ha': 'left', 'va': 'bottom', 'fontsize': 22},
        {'x': 0.03, 'y': 0.65, 'text': rf'$k = {ks[1]}$', 'ha': 'left', 'va': 'bottom', 'fontsize': 22},
        {'x': 0.03, 'y': 0.65, 'text': rf'$k = {ks[2]}$', 'ha': 'left', 'va': 'bottom', 'fontsize': 22},
    ]

    save_uncover_multiaxes_slides(
        tsavef,
        y_groups,
        ['FOM', 'POD-Gal.', 'OpInf', 'GasOpInf', 'NiTROM', 'GasNiTROM'],
        'figures/slides/cavity_forcing_0p1_energy_uncover',
        colors=colors,
        linestyles=lstyle,
        linewidths=[2, 2, 2, 2, 2, 2],
        ncols=1,
        xlabels=[None, None, r'Time $t$'],
        ylabels=[None, 'Energy', None],
        titles=[rf'$w(t) = {amp} \sin(kt)$', None, None],
        xlims=[(0.0, tsavef[-1]), (0.0, tsavef[-1]), (0.0, tsavef[-1])],
        ylims=ylims,
        texts=texts,
        hide_xticks_except_last=True,
        legend_axis_index=0,
        legend_loc='upper right',
        fixed_legend_size=True,
        show_legend=False,
        save_eps=True,
    )


# Snapshot contour plots
idx = np.argmin(np.abs(tsavef - 30))
ii = 2

X, Y, fields = pp.output_fields(flow, dataf[:, idx] - flow.q_sbf)
vmin = np.min(fields[ii])
vmax = -vmin
print(vmin, vmax)

# Build all snapshot states first
snapshots = [
    ("FOM", dataf[:, idx] - flow.q_sbf),
    ("POD-Gal.", phi_pre @ sol_pod.cpu().numpy()[:, idx]),
    ("OpInf", phi_pre @ sol_oi.cpu().numpy()[:, idx]),
    ("GasOpInf", phi_pre @ sol_oi_gs.cpu().numpy()[:, idx]),
    ("NiTROM", phi_pre @ sol_nit.cpu().numpy()[:, idx]),
    ("GasNiTROM", phi_pre @ sol_nit_gs.cpu().numpy()[:, idx]),
]

fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH_WIDE, 5.0), constrained_layout=True)
axes = axes.ravel()

for idx_subplot, (ax, (title, state_vec)) in enumerate(zip(axes, snapshots)):
    X, Y, fields = pp.output_fields(flow, state_vec)
    # if title == "NiTROM":
    #     fields[ii] = np.zeros_like(fields[ii])
    #     title += " (blew up)"
    cf = ax.contourf(
        X[ii][:39, :],
        Y[ii][:39, :],
        np.flipud(fields[ii])[:39, :],
        levels=100,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax
    )
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='out', top=False, right=False)
    
    # Only show x-axis labels on bottom row (indices 4, 5)
    if idx_subplot < 4:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x$')
    
    # Only show y-axis on left column (indices 0, 2, 4)
    if idx_subplot % 2 != 0:
        ax.set_yticks([])
    else:
        ax.set_ylabel(r'$y$')
    
    ax.text(
        0.5, 0.65, title,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=20,
    )

# fig.colorbar(cf, ax=axes.tolist(), shrink=0.92, pad=0.02, label='Vorticity')
save_figure(fig, "cavity_50_forcing_0p1_k4_snapshot_all")


# # Truth
# print(vmin, vmax)
# plt.figure()
# plt.contourf(X[ii][:39, :], Y[ii][:39, :], np.flipud(fields[ii])[:39, :], levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.text(
#     0.5, 0.65, 'FOM',
#     transform=ax.transAxes,
#     ha='center', va='bottom',
#     fontsize=33,
# )
# plt.tight_layout()
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_truth.eps", format='eps', bbox_inches='tight')
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_truth", bbox_inches='tight')

# # POD
# X, Y, fields = pp.output_fields(flow, phi_pre @ sol_pod.cpu().numpy()[:, idx])
# plt.figure()
# plt.contourf(X[ii][:39, :], Y[ii][:39, :], np.flipud(fields[ii])[:39, :], levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.text(
#     0.5, 0.65, 'POD-Gal.',
#     transform=ax.transAxes,
#     ha='center', va='bottom',
#     fontsize=33,
# )
# plt.tight_layout()
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_pod.eps", format='eps', bbox_inches='tight')
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_pod", bbox_inches='tight')

# # OpInf
# X, Y, fields = pp.output_fields(flow, phi_pre @ sol_oi.cpu().numpy()[:, idx])
# plt.figure()
# plt.contourf(X[ii][:39, :], Y[ii][:39, :], np.flipud(fields[ii])[:39, :], levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.text(
#     0.5, 0.65, 'OpInf',
#     transform=ax.transAxes,
#     ha='center', va='bottom',
#     fontsize=33,
# )
# plt.tight_layout()
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_opinf.eps", format='eps', bbox_inches='tight')
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_opinf", bbox_inches='tight')

# # GasOpInf
# X, Y, fields = pp.output_fields(flow, phi_pre @ sol_oi_gs.cpu().numpy()[:, idx])
# plt.figure()
# plt.contourf(X[ii][:39, :], Y[ii][:39, :], np.flipud(fields[ii])[:39, :], levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.text(
#     0.5, 0.65, 'GasOpInf',
#     transform=ax.transAxes,
#     ha='center', va='bottom',
#     fontsize=33,
# )
# plt.tight_layout()
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_gasopinf.eps", format='eps', bbox_inches='tight')
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_gasopinf", bbox_inches='tight')

# # NiTROM
# X, Y, fields = pp.output_fields(flow, phi_pre @ sol_nit.cpu().numpy()[:, idx])
# plt.figure()
# plt.contourf(X[ii][:39, :], Y[ii][:39, :], np.flipud(fields[ii])[:39, :], levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.text(
#     0.5, 0.65, 'NiTROM',
#     transform=ax.transAxes,
#     ha='center', va='bottom',
#     fontsize=33,
# )
# plt.tight_layout()
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_nitrom.eps", format='eps', bbox_inches='tight')
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_nitrom", bbox_inches='tight')

# # GasNiTROM
# X, Y, fields = pp.output_fields(flow, phi_pre @ sol_nit_gs.cpu().numpy()[:, idx])
# plt.figure()
# plt.contourf(X[ii][:39, :], Y[ii][:39, :], np.flipud(fields[ii])[:39, :], levels=100, cmap='bwr', vmin=vmin, vmax=vmax)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.text(
#     0.5, 0.65, 'GasNiTROM',
#     transform=ax.transAxes,
#     ha='center', va='bottom',
#     fontsize=33,
# )
# plt.tight_layout()
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_gasnitrom.eps", format='eps', bbox_inches='tight')
# plt.savefig("figures/cavity_forcing_0p7_k2_snapshot_gasnitrom", bbox_inches='tight')
