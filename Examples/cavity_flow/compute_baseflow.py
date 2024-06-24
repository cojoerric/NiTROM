#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:43:38 2023

@author: alberto
"""


import matplotlib.pyplot as plt
import numpy as np 
import time_steppers as tstep
import post_process as pp
import classes_cavity as classes


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

flow = classes.flow_parameters(Lx,Ly,Nx,Ny,Re)

n = 400
dt = 1.0/n

lops = classes.linear_operators_2D(flow,dt)

#%% Compute base flow
nsave = 1000
time = dt*np.arange(0,n*800,1)
q0 = np.zeros(flow.szu+flow.szv)
# q0 = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))

# q0 = data[:,-1].copy()
data, tsave = tstep.nonlinear_solver_2D(flow,lops,q0,time,nsave)

#%%
idx0 = 0
idx1 = data.shape[-1]

energy = pp.compute_energy(data[:,idx0:idx1],data[:,0],tsave[idx0:idx1])

plt.figure()
plt.plot(tsave[idx0:idx1]-tsave[idx0],energy)
plt.plot(tsave[idx0:idx1]-tsave[idx0],energy,'rx')

#%%
np.save("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny),data[:,-1])

#%%
# idxpl = np.argmin(np.abs(tsave - 8*flow.T/10))
# idxpl = 0
ii = -2
# mean = np.mean(data[:flow.szu+flow.szv],axis=-1)
X, Y, fields = pp.output_fields(flow,data[:,-1].real)

color_map = plt.cm.get_cmap('bwr')

idx = 2
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


