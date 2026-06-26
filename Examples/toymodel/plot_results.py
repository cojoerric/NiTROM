import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from mpi4py import MPI

from NiTROM.Optimization_Functions import classes
import fom_class


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
lPOD, lOI, lOI_gs, lNIT, lNIT_gs = (0, (1.0, 1.2)), (0, (5.0, 2.4)), "solid", (0, (8.0, 2.5)), "solid"


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
        bbox=dict(facecolor="white", edgecolor="#6f6f6f", linewidth=0.6, alpha=0.95, boxstyle="round,pad=0.25"),
    )


def save_figure(fig, stem):
    fig.savefig(f"figures/{stem}.eps", format="eps")
    # fig.savefig(f"figures/{stem}.pdf", format="pdf")
    fig.savefig(f"figures/{stem}.png", format="png")
    plt.close(fig)



n = 3
n_traj = 4
beta = 20.0
diag_vec = np.array([-1.0,-2.0,-5.0])
A2 = np.diag(diag_vec)
A3 = np.zeros((3,3,3))
diag_vec2 = np.array([beta,beta,0.0])
A3[:,:,-1] = np.diag(diag_vec2)
B = np.ones((3,1))
C = np.ones((1,3))
fom = fom_class.full_order_model(A2,A3,B,C)

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_steady_forcing':fname_forcing,
               'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

r = 2               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

#%% Compute NiTROM model 

which_trajs = np.arange(0,pool.n_traj)
which_times = np.arange(0,pool.n_snapshots)
leggauss_deg = 5
nsave_rom = 10

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

N = pool.n_snapshots*pool.n_traj
X = np.zeros((pool.X.shape[1],N))
for i in range (pool.n_traj):
    X[:,i*pool.n_snapshots:(i+1)*pool.n_snapshots] = pool.X[i,]
phi_pod, _, _ = np.linalg.svd(X,full_matrices=False)
phi_pod = phi_pod[:,:r]
psi_pod = phi_pod.copy()

A_pod = np.load('results/A_pod.npy')
H_pod = np.load('results/H_pod.npy')
tensors_pod = (A_pod, H_pod)

A_oi = np.load('results/A_oi.npy')
H_oi = np.load('results/H_oi.npy')
tensors_oi = (A_oi, H_oi)

A_oi_gs = np.load('results/A_oi_gs.npy')
H_oi_gs = np.load('results/H_oi_gs.npy')
tensors_oi_gs = (A_oi_gs, H_oi_gs)

phi_nit = np.load('results/phi_nit.npy')
psi_nit = np.load('results/psi_nit.npy')
A_nit = np.load('results/A_nit.npy')
H_nit = np.load('results/H_nit.npy')
tensors_nit = (A_nit, H_nit)

phi_nit_gs = np.load('results/phi_nit_gs.npy')
psi_nit_gs = np.load('results/psi_nit_gs.npy')
A_nit_gs = np.load('results/A_nit_gs.npy')
H_nit_gs = np.load('results/H_nit_gs.npy')
tensors_nit_gs = (A_nit_gs, H_nit_gs)

# Plot errors
max_val = 5/20
betas = np.random.rand(100)*0.999*max_val
# betas = np.asarray([0.01,0.1,0.2,0.248])
t_eval = np.linspace(0, 30, 300)
idx_10 = np.argmin(np.abs(t_eval - 10.0))
error_pod = np.zeros_like(t_eval)
error_oi = np.zeros_like(t_eval)
error_oi_gs = np.zeros_like(t_eval)
error_nit = np.zeros_like(t_eval)
error_nit_gs = np.zeros_like(t_eval)

for k in range(len(betas)):
    u = betas[k]*np.ones(n)
    x0 = np.zeros(n)
    z0 = np.zeros(r)

    sol = solve_ivp(fom.evaluate_fom_dynamics,[0,t_eval[-1]],x0,'RK45',t_eval=t_eval,args=(u,)).y
    id_ss = np.array([-betas[k]/(-1 + 4*betas[k]), -betas[k]/(-2 + 4*betas[k]), betas[k]/5])
    weight = np.linalg.norm(fom.compute_output(id_ss))**2

    sol_pod_r = solve_ivp(opt_obj.evaluate_rom_rhs,[0,t_eval[-1]],z0,'RK45',t_eval=t_eval,args=(psi_pod.T@u,) + tensors_pod).y
    sol_pod = phi_pod @ sol_pod_r
    error_pod += np.linalg.norm(C @ (sol_pod - sol), axis=0)**2 / weight / len(betas)

    sol_oi_r = solve_ivp(opt_obj.evaluate_rom_rhs,[0,t_eval[-1]],z0,'RK45',t_eval=t_eval,args=(psi_pod.T@u,) + tensors_oi).y
    sol_oi = phi_pod @ sol_oi_r
    error_oi += np.linalg.norm(C @ (sol_oi - sol), axis=0)**2 / weight / len(betas)

    sol_oi_gs_r = solve_ivp(opt_obj.evaluate_rom_rhs,[0,t_eval[-1]],z0,'RK45',t_eval=t_eval,args=(psi_pod.T@u,) + tensors_oi_gs).y
    sol_oi_gs = phi_pod @ sol_oi_gs_r
    error_oi_gs += np.linalg.norm(C @ (sol_oi_gs - sol), axis=0)**2 / weight / len(betas)

    sol_nit_r = solve_ivp(opt_obj.evaluate_rom_rhs,[0,t_eval[-1]],z0,'RK45',t_eval=t_eval,args=(psi_nit.T@u,) + tensors_nit).y
    sol_nit = phi_nit @ sol_nit_r
    error_nit += np.linalg.norm(C @ (sol_nit - sol), axis=0)**2 / weight / len(betas)

    sol_nit_gs_r = solve_ivp(opt_obj.evaluate_rom_rhs,[0,t_eval[-1]],z0,'RK45',t_eval=t_eval,args=(psi_nit_gs.T@u,) + tensors_nit_gs).y
    sol_nit_gs = phi_nit_gs @ sol_nit_gs_r
    error_nit_gs += np.linalg.norm(C @ (sol_nit_gs - sol), axis=0)**2 / weight / len(betas)

fig, ax = make_figure()
ax.semilogy(t_eval[:idx_10], error_pod[:idx_10], label='POD-Gal.', color=cPOD, linestyle=lPOD)
ax.semilogy(t_eval[:idx_10], error_oi[:idx_10], label='OpInf', color=cOI, linestyle=lOI)
ax.semilogy(t_eval[:idx_10], error_oi_gs[:idx_10], label='GasOpInf', color=cOI_gs, linestyle=lOI_gs)
ax.semilogy(t_eval[:idx_10], error_nit[:idx_10], label='NiTROM', color=cNIT, linestyle=lNIT)
ax.semilogy(t_eval[:idx_10], error_nit_gs[:idx_10], label='GasNiTROM', color=cNIT_gs, linestyle=lNIT_gs)
style_axes(ax, xlabel='Time $t$', ylabel='Average error $e(t)$', xlim=(0.0, 10.0), log_y=True)
save_figure(fig, 'error_toymodel_10')

fig, ax = make_figure(wide=True)
ax.semilogy(t_eval, error_pod, label='POD-Gal.', color=cPOD, linestyle=lPOD)
ax.semilogy(t_eval, error_oi, label='OpInf', color=cOI, linestyle=lOI)
ax.semilogy(t_eval, error_oi_gs, label='GasOpInf', color=cOI_gs, linestyle=lOI_gs)
ax.semilogy(t_eval, error_nit, label='NiTROM', color=cNIT, linestyle=lNIT)
ax.semilogy(t_eval, error_nit_gs, label='GasNiTROM', color=cNIT_gs, linestyle=lNIT_gs)
style_axes(ax, xlabel='Time $t$', ylabel='Average error $e(t)$', xlim=(0.0, 30.0), ylim=(None, 1e1), log_y=True)
add_training_window(ax)
ax.legend(loc='lower right', ncol=2, columnspacing=1.0, handletextpad=0.5)
save_figure(fig, 'error_toymodel_30')



tk = np.linspace(0, 30, 1000)
uk = 0.45*(np.sin(tk) + np.cos(2*tk))

t_eval = np.linspace(0, 30, 1000)
idx_10 = np.argmin(np.abs(tk - 10.0))

fu = interp1d(tk, np.outer(B.flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_fom = solve_ivp(fom.evaluate_fom_dynamics, [0, tk[-1]], np.zeros(3), 'RK45', t_eval=tk, args=(fu,)).y

fu = interp1d(tk, np.outer((psi_pod.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_pod = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_pod).y

fu = interp1d(tk, np.outer((psi_pod.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_oi = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_oi).y
sol_oi_gs = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_oi_gs).y

fu = interp1d(tk, np.outer((psi_nit.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_nit = phi_nit @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_nit).y

fu = interp1d(tk, np.outer((psi_nit_gs.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_nit_gs = phi_nit_gs @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_nit_gs).y

fig, ax = make_figure()
ax.plot(t_eval[:idx_10], fom.compute_output(sol_fom)[0,][:idx_10], color='k', linewidth=2.4, label='FOM')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_pod)[0,][:idx_10], color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_nit)[0,][:idx_10], color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_nit_gs)[0,][:idx_10], color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_oi)[0,][:idx_10], color=cOI, linestyle=lOI, label='OpInf')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_oi_gs)[0,][:idx_10], color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
style_axes(ax, xlabel='Time $t$', ylabel='$y(t)$', xlim=(0.0, 10.0))
save_figure(fig, 'response_toymodel_low_10')

fig, ax = make_figure(wide=True)
ax.plot(t_eval, fom.compute_output(sol_fom)[0,], color='k', linewidth=2.4, label='FOM')
ax.plot(t_eval, fom.compute_output(sol_pod)[0,], color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(t_eval, fom.compute_output(sol_oi)[0,], color=cOI, linestyle=lOI, label='OpInf')
ax.plot(t_eval, fom.compute_output(sol_nit)[0,], color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(t_eval, fom.compute_output(sol_oi_gs)[0,], color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
ax.plot(t_eval, fom.compute_output(sol_nit_gs)[0,], color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
style_axes(ax, xlabel='Time $t$', ylabel='$y(t)$', xlim=(0.0, 30.0))
# add_training_window(ax)
save_figure(fig, 'response_toymodel_low_30')



tk = np.linspace(0, 30, 1000)
idx_10 = np.argmin(np.abs(tk - 10.0))
uk = 0.65*(np.sin(tk) + np.cos(2*tk))

t_eval = np.linspace(0, 30, 1000)

fu = interp1d(tk, np.outer(B.flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_fom = solve_ivp(fom.evaluate_fom_dynamics, [0, tk[-1]], np.zeros(3), 'RK45', t_eval=tk, args=(fu,)).y

fu = interp1d(tk, np.outer((psi_pod.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_pod = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_pod).y

fu = interp1d(tk, np.outer((psi_pod.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_oi = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_oi).y
sol_oi_gs = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_oi_gs).y

fu = interp1d(tk, np.outer((psi_nit.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_nit = phi_nit @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_nit).y

fu = interp1d(tk, np.outer((psi_nit_gs.T@B).flatten(), uk), kind='cubic', bounds_error=False, fill_value='extrapolate')
sol_nit_gs = phi_nit_gs @ solve_ivp(opt_obj.evaluate_rom_rhs, [0, tk[-1]], np.zeros(r), 'RK45', t_eval=tk, args=(fu,) + tensors_nit_gs).y

fig, ax = make_figure()
ax.plot(t_eval[:idx_10], fom.compute_output(sol_fom)[0,][:idx_10], color='k', linewidth=2.4, label='FOM')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_pod)[0,][:idx_10], color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_nit)[0,][:idx_10], color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_nit_gs)[0,][:idx_10], color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_oi)[0,][:idx_10], color=cOI, linestyle=lOI, label='OpInf')
ax.plot(t_eval[:idx_10], fom.compute_output(sol_oi_gs)[0,][:idx_10], color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
style_axes(
    ax,
    xlabel='Time $t$',
    ylabel='$y(t)$',
    xlim=(0.0, 10.0),
    ylim=(
        fom.compute_output(sol_fom)[:,:idx_10].min() * 2,
        fom.compute_output(sol_fom)[:,:idx_10].max() * 2.5,
    ),
)
save_figure(fig, 'response_toymodel_high_10')

fig, ax = make_figure(wide=True)
ax.plot(t_eval, fom.compute_output(sol_fom)[0,], color='k', linewidth=2.4, label='FOM')
ax.plot(t_eval, fom.compute_output(sol_pod)[0,], color=cPOD, linestyle=lPOD, label='POD-Gal.')
ax.plot(t_eval, fom.compute_output(sol_nit)[0,], color=cNIT, linestyle=lNIT, label='NiTROM')
ax.plot(t_eval, fom.compute_output(sol_nit_gs)[0,], color=cNIT_gs, linestyle=lNIT_gs, label='GasNiTROM')
ax.plot(t_eval, fom.compute_output(sol_oi)[0,], color=cOI, linestyle=lOI, label='OpInf')
ax.plot(t_eval, fom.compute_output(sol_oi_gs)[0,], color=cOI_gs, linestyle=lOI_gs, label='GasOpInf')
style_axes(
    ax,
    xlabel='Time $t$',
    ylabel='$y(t)$',
    xlim=(0.0, 30.0),
    ylim=(fom.compute_output(sol_fom).min() * 2, 12.0),
)
# add_training_window(ax)
save_figure(fig, 'response_toymodel_high_30')