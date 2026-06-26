import numpy as np
import scipy as sp
import scipy.interpolate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from mpi4py import MPI

import time_steppers as tstep
import post_process as pp

from NiTROM.Optimization_Functions import classes
import classes_cavity

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
    fig.savefig(f"figures/{stem}.png", format="png")
    plt.close(fig)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

if rank == 0:
    print(f"Using {world_size} MPI process(es) for CPU figure generation.")

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
n_traj = len(amps)
n = phi_pre.shape[-1]

pool_inputs = (comm, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

r = 50               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 15

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

phi_pod = np.eye(n, r)
psi_pod = phi_pod.copy()
A_pod = np.load('results/A_pod.npy')
H_pod = np.load('results/H_pod.npy')
tensors_pod = (A_pod, H_pod)

A_oi = np.load('results/A_oi.npy')
H_oi = np.load('results/H_oi.npy').reshape(r,r,r)
tensors_oi = (A_oi, H_oi)

A_oi_gs = np.load('results/A_oi_gs.npy')
H_oi_gs = np.load('results/H_oi_gs.npy')
tensors_oi_gs = (A_oi_gs, H_oi_gs)

phi_nit = np.load('results/phi_nit.npy')
psi_nit = np.load('results/psi_nit.npy')
A_nit = np.load('results/A_nit.npy')
H_nit = np.load('results/H_nit.npy').reshape(r,r,r)
tensors_nit = (A_nit, H_nit)

phi_nit_gs = np.load('results/phi_nit_gs.npy')
psi_nit_gs = np.load('results/psi_nit_gs.npy')
A_nit_gs = np.load('results/A_nit_gs.npy')
H_nit_gs = np.load('results/H_nit_gs.npy')
tensors_nit_gs = (A_nit_gs, H_nit_gs)


## Plot results
# Energy of perturbations
if which == 'train':
    fig, ax = make_figure(wide=False)
    for k in range (pool.my_n_traj):
        Qk = pool.X[k,]
        energy_k = np.linalg.norm(Qk, axis=0)**2
        ax.plot(pool.time, energy_k, color='k', alpha=0.85)
    style_axes(ax, xlabel=r'Time $t$', ylabel='Energy of perturbations', xlim=(0, pool.time[-1]))
    ax.set_ylim(bottom=0)
    add_training_window(ax, text_xy=(0.25, 0.78))
    save_figure(fig, "cavity_energy_perturbations")


# Testing/training error
time = pool.time
u = np.zeros(r)
e_pod = 0
e_oi = 0
e_oi_gs = 0
e_nit = 0
e_nit_gs = 0

idx_final = 80
fig, ax = make_figure(wide=False)
for k in range(n_traj):
    mean_en = np.mean(np.linalg.norm(pool.X[k,], axis=0)**2)

    # POD
    z_pod = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_pod, method='RK45', t_eval=time, args=(u,) + tensors_pod).y
    e_pod += np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en / n_traj

    # OpInf
    z_oi = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_oi, method='RK45', t_eval=time, args=(u,) + tensors_oi).y
    e_oi += np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en / n_traj

    # OpInf GS
    z_oi_gs = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_oi_gs, method='RK45', t_eval=time, args=(u,) + tensors_oi_gs).y
    e_oi_gs += np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en / n_traj

    # NiTROM
    z_nit = psi_nit.T @ pool.X[k,:,0]
    sol = phi_nit @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_nit, method='RK45', t_eval=time, args=(u,) + tensors_nit).y
    e_nit += np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en / n_traj

    # NiTROM GS
    z_nit_gs = psi_nit_gs.T @ pool.X[k,:,0]
    sol = phi_nit_gs @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_nit_gs, method='RK45', t_eval=time, args=(u,) + tensors_nit_gs).y
    e_nit_gs += np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en / n_traj

ax.plot(time, e_pod, color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(time, e_oi, color=cOI, linestyle=lOI, label='OpInf')
ax.plot(time, e_oi_gs, color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
ax.plot(time, e_nit, color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(time, e_nit_gs, color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
style_axes(ax, xlabel='Time $t$', ylabel='Error', xlim=(0, pool.time[idx_final]), ylim=(1e-3, None), log_y=True)
save_figure(fig, f'cavity_50_error_{which}_trained')

fig, ax = make_figure(wide=True)
ax.plot(time, e_pod, color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(time, e_oi, color=cOI, linestyle=lOI, label='OpInf')
ax.plot(time, e_oi_gs, color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
ax.plot(time, e_nit, color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(time, e_nit_gs, color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
style_axes(ax, xlabel='Time $t$', ylabel='Error', xlim=(0, pool.time[-1]), ylim=(1e-3, None), log_y=True)
add_training_window(ax)
ax.legend(loc='upper right', ncol=3, columnspacing=1.0, handletextpad=0.5)
save_figure(fig, f'cavity_50_error_{which}_full')


# Sinusoidal forcing
time_np = dt*np.arange(0, 80*n, 1)
nsave = 5
amp = 0.7
energies = []
ks = [1,2,4]

for (k, harmonic) in enumerate(ks):
    freq = 1.00 * harmonic
    tf = np.arange(0, 2*np.pi/freq, dt)
    fint = sp.interpolate.interp1d(tf, amp * np.sin(freq * tf), kind='linear', fill_value="extrapolate")
    if rank == 0:
        print("Forcing frequency %.2f..." % freq)

    qic = flow.q_sbf.copy()
    dataf, tsavef = tstep.solver_2D(flow,lops,qic,time_np,nsave,[0,1,1,0,0,0,0,0],[1],[fint],[2*np.pi/freq],vol_forcing=B)
    energy_true = np.linalg.norm(dataf - qic.reshape(-1,1), dim=0)**2

    z0 = np.zeros(r)
    t_eval_sin = time_np[::nsave]

    # POD
    B_proj_pod = psi_pod.T @ phi_pre.T @ B
    fpod = lambda t: B_proj_pod * amp * np.sin(freq * t)
    sol_pod = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [t_eval_sin[0], t_eval_sin[-1]], z0, method='RK45', t_eval=t_eval_sin, args=(fpod,) + tensors_pod).y
    energy_pod = np.linalg.norm(sol_pod, axis=0)**2

    # OpInf
    B_proj_oi = psi_pod.T @ phi_pre.T @ B
    foi = lambda t: B_proj_oi * amp * np.sin(freq * t)
    sol_oi = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [t_eval_sin[0], t_eval_sin[-1]], z0, method='RK45', t_eval=t_eval_sin, args=(foi,) + tensors_oi).y
    energy_oi = np.linalg.norm(sol_oi, axis=0)**2

    # GsOpInf
    B_proj_oi_gs = psi_pod.T @ phi_pre.T @ B
    foi_gs = lambda t: B_proj_oi_gs * amp * np.sin(freq * t)
    sol_oi_gs = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [t_eval_sin[0], t_eval_sin[-1]], z0, method='RK45', t_eval=t_eval_sin, args=(foi_gs,) + tensors_oi_gs).y
    energy_oi_gs = np.linalg.norm(sol_oi_gs, axis=0)**2

    # NiTROM
    B_proj_nit = psi_nit.T @ phi_pre.T @ B
    fnit = lambda t: B_proj_nit * amp * np.sin(freq * t)
    sol_nit = phi_nit @ solve_ivp(opt_obj.evaluate_rom_rhs, [t_eval_sin[0], t_eval_sin[-1]], z0, method='RK45', t_eval=t_eval_sin, args=(fnit,) + tensors_nit).y
    energy_nit = np.linalg.norm(sol_nit, axis=0)**2

    # GsNiTROM
    B_proj_nit_gs = psi_nit_gs.T @ phi_pre.T @ B
    fnit_gs = lambda t: B_proj_nit_gs * amp * np.sin(freq * t)
    sol_nit_gs = phi_nit_gs @ solve_ivp(opt_obj.evaluate_rom_rhs, [t_eval_sin[0], t_eval_sin[-1]], z0, method='RK45', t_eval=t_eval_sin, args=(fnit_gs,) + tensors_nit_gs).y
    energy_nit_gs = np.linalg.norm(sol_nit_gs, axis=0)**2

    energy_lst = [energy_true, energy_pod, energy_oi, energy_oi_gs, energy_nit, energy_nit_gs]
    energies.append(energy_lst)

colors = ['k', cPOD, cOI, cOI_gs, cNIT, cNIT_gs]
lstyle = ['-', lPOD, lOI, lOI_gs, lNIT, lNIT_gs]
fig, ax = plt.subplots(nrows=3, ncols=1)
for k in range(len(energies)):
    for (i, vec) in enumerate(energies[k]):
        ax[k].plot(tsavef, vec, color=colors[i], linestyle=lstyle[i])
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
        ax[k].set_ylim(0, energies[k][-2].max()*1.1)
    else:
        ax[k].set_ylim(0, energies[k][-1].max()*1.1)
    style_axes(ax[k], xlim=(0, tsavef[-1]), ylim=(0, ax[k].get_ylim()[1]))
    ax[k].grid(which="minor", visible=False)

ax[-1].set_xlabel(r'Time $t$')
ax[1].set_ylabel('Energy')
fig.set_size_inches(FIG_WIDTH_WIDE, 6.6)
save_figure(fig, "cavity_50_forcing_0p7_energy")


# Snapshot contour plots
idx = np.argmin(np.abs(tsavef - 30))
ii = 2

X, Y, fields = pp.output_fields(flow, dataf[:, idx] - flow.q_sbf)
vmin = np.min(fields[ii])
vmax = -vmin
if rank == 0:
    print(vmin, vmax)

# Build all snapshot states first
snapshots = [
    ("FOM", dataf[:, idx] - flow.q_sbf),
    ("POD-Gal.", phi_pre @ sol_pod[:, idx]),
    ("OpInf", phi_pre @ sol_oi[:, idx]),
    ("GasOpInf", phi_pre @ sol_oi_gs[:, idx]),
    ("NiTROM", phi_pre @ sol_nit[:, idx]),
    ("GasNiTROM", phi_pre @ sol_nit_gs[:, idx]),
]

fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH_WIDE, 5.0), constrained_layout=True)
axes = axes.ravel()

for idx_subplot, (ax, (title, state_vec)) in enumerate(zip(axes, snapshots)):
    X, Y, fields = pp.output_fields(flow, state_vec)
    if title == "NiTROM":
        fields[ii] = np.zeros_like(fields[ii])
        title += " (blew up)"
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
    
    if idx_subplot < 4:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x$')
    
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

save_figure(fig, "cavity_50_forcing_0p7_k4_snapshot_all")
