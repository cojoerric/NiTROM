#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:40:38 2024

@author: alberto
"""



import matplotlib.pyplot as plt
import numpy as np 
import time_steppers as tstep
import post_process as pp
import classes_cavity as classes
import numba_operators as numbaops


plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],
                     "font.size":12})

#%%

Lx = 1
Ly = 1
Nx = 100
Ny = 100

dx = Lx/Nx
dy = Ly/Ny
Re = 8300


flow = classes.flow_class(Lx,Ly,Nx,Ny,Re)

n = 400
dt = 1.0/n

lops = classes.linear_operators_2D(flow,dt)

#%%
nsave = 1000
time = dt*np.arange(0,n*400,1)
q0 = np.zeros(flow.szu+flow.szv)

bc_coefs = [0,0,1,0,0,0,0,0] # ul, ur, ub, ut, vl, vr, vb, vt

# q0 = data[:,-1]
q0 = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
data, tsave = tstep.solver_2D(flow,lops,q0,time,n,bc_coefs)


#%%
idx0 = 0
idx1 = data.shape[-1]

energy = pp.compute_energy(data[:,idx0:idx1],data[:,0],tsave[idx0:idx1])

plt.figure()
plt.plot(tsave[idx0:idx1]-tsave[idx0],energy)
plt.plot(tsave[idx0:idx1]-tsave[idx0],energy,'rx')

#%%
# np.save("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny),data[:,-1])

#%%
# idxpl = np.argmin(np.abs(tsave - 8*flow.T/10))
# idxpl = 0

# mean = np.mean(data[:flow.szu+flow.szv],axis=-1)
X, Y, fields = pp.output_fields(flow,flow.q_sbf)

color_map = plt.cm.get_cmap('bwr')

idx = 1
vmin = np.min(fields[idx]) 
vmax = -vmin

plt.figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap=color_map,vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()
# ax.set_xticks([0,0.25,0.5,0.75,1.0])
# ax.set_yticks([0,0.25,0.5,0.75,1.0])

# plt.savefig("./Figures/bflow_Re%d.eps"%Re,format='eps')
# plt.savefig("./Figures/bflow_Re%d.png"%Re)

u = fields[0]
v = fields[1]

print(np.max(fields[idx]),np.min(fields[idx]))

#%% ERASE ALL THIS BELOW

flow_pert = classes.flow_class(Lx,Ly,Nx,Ny,Re)
flow_pert.create_side_wall_forcing_profile(0.45,0.55,'ur')

flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
flow_pert.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))

fom = classes.fom_class(flow,lops)


#%%

# plt.figure()
# plt.plot(flow.y,flow.ur)

#%%

# Compute the effect of the BC on the flow (then reset the flow class)
flow.create_side_wall_forcing_profile(0.45,0.55,'ur')
fom = classes.fom_class(flow,lops)
B = fom.evaluate_full_fom_dynamics(np.zeros(flow.szu + flow.szv),[0,1e-4,1,0,0,0,0,0])/1e-4
flow.ur *= 0
fom = classes.fom_class(flow,lops)

print(np.linalg.norm(B))

#%%

import time as tlib

nsave = 100
time = dt*np.arange(0,n*40,1)
tsave = time[::nsave]

amps = np.asarray([1e-1])


Q = np.zeros((flow.szu + flow.szv,len(amps)*len(tsave)))
energy = np.zeros((len(amps),len(tsave)))

for k in range (len(amps)):
    
    t0 = tlib.time()
    print("Generating trajectory %d/%d"%(k+1,len(amps)))
    qic = flow.q_sbf + amps[k]*B
    data, _ = tstep.solver_2D(flow,lops,qic,time,nsave,bc_coefs)
    data -= flow.q_sbf.reshape(-1,1)
    
    Q[:,k*len(tsave):(k+1)*len(tsave)] = data 
    energy[k,] = np.linalg.norm(data,axis=0)**2
    t1 = tlib.time() - t0
    print("Execution time = %1.3f [min]"%(t1/60))
    
#%%
import scipy
eps = 0.01
freq = 4
tf = np.arange(0,2*np.pi/freq,dt)
time = dt*np.arange(0,n*40,1)

nsave = 50
flow.create_side_wall_forcing_profile(0.45,0.55,'ur')
fint = scipy.interpolate.interp1d(tf,eps*np.sin(freq*tf),kind='linear',fill_value='extrapolate')
qic = flow.q_sbf.copy()
# data2, tsavef = tstep.solver_2D(flow,lops,qic,time,nsave,[0,1,1,0,0,0,0,0],[1],[fint],[2*np.pi/freq])

flow.ur *= 0
dataf, tsavef = tstep.solver_2D(flow,lops,qic,time,nsave,[0,1,1,0,0,0,0,0],[1],[fint],[2*np.pi/freq],vol_forcing=B)

# bc_coefs = [0,0,1,0,0,0,0,0]
# data2, _ = tstep.solver_2D(flow,lops,flow.q_sbf,time,nsave,bc_coefs)


energy_true = np.linalg.norm(data2 - flow.q_sbf.reshape(-1,1),axis=0)**2
energy_true2 = np.linalg.norm(dataf - flow.q_sbf.reshape(-1,1),axis=0)**2

#%%

idx0 = 0
idx1 = data2.shape[-1]

# energy = pp.compute_energy(data2[:,idx0:idx1],data2[:,0],tsavef[idx0:idx1])

plt.figure()
# plt.plot(tsavef[idx0:idx1]-tsavef[idx0],energy)
# plt.plot(tsavef[idx0:idx1]-tsavef[idx0],energy,'rx')
plt.plot(tsavef[idx0:idx1]-tsavef[idx0],energy_true)
plt.plot(tsavef[idx0:idx1]-tsavef[idx0],energy_true2)

#%%

plt.figure()
for k in range (len(amps)):
    plt.plot(tsave,energy[k,],'k')
    
# plt.gca().set_ylim(0,50)

#%%

import scipy.linalg as sciplin
U, S, _ = sciplin.svd(Q,full_matrices=False)
Slo = 1 - np.cumsum(S**2)/np.sum(S**2)

#%%
plt.figure()
plt.plot(Slo,'o')

ax = plt.gca()
ax.set_yscale('log')

#%%

Phi = U[:,:10]
Psi = Phi.copy()

#%%
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
fom = classes.fom_class(flow,lops)
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi,Psi,B,bc_coefs)
A2, A3 = tensors_pod[0], tensors_pod[1]


vec = np.random.randn(flow.szu + flow.szv)
z = Psi.T@vec
vec = Phi@z

fbflow = fom.evaluate_full_fom_dynamics(0*flow.q_sbf, bc_coefs)
ftrue = fom.evaluate_fom_dynamics(vec,bc_coefs,np.zeros(8)) + fbflow
ffull = fom.evaluate_full_fom_dynamics(vec, bc_coefs)


print(np.linalg.norm(fbflow))
print(np.linalg.norm(ftrue - ffull))


fz_ = Psi.T@ftrue
fz = A2@z + np.einsum('ijk,j,k',A3,z,z)

print(np.linalg.norm(fz - fz_))


#%%





