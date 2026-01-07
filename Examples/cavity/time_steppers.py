#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:02:38 2022

@author: alberto
"""
import numpy as np
import numba_operators as numbaops
import time as tlib

def nonlinear_solver_2D(flow,lops,qic,time,n,*argv):
    
    q = qic.copy()
    tsave = time[::n]
    data = np.zeros((len(q),len(tsave)))
    
    
    data[:,0] =  q
    
    # Initialize vectors for time stepping
    qrhs = np.zeros(len(q))
    qfwd = qrhs.copy()
    qkm1 = qrhs.copy()
    qs = qrhs.copy()
    p = np.zeros(lops.G.shape[-1])
    
    # Initialize vectors for boundary conditions contribution
    qtop = np.zeros(len(flow.x)-1)
    qlap_bc = np.zeros(len(q))
    
    if len(argv) > 0: 
        fu = argv[0]    # interp1d interpolator
        tu = argv[1]    # period of forcing
        flag = 1
    else:
        flag = 0
        
    
    idx_save = 1
    ttot = 0
    for k in range (1,len(time)):
        
        t0 = tlib.perf_counter()
        
        if flag == 0:   ff = np.zeros(len(q))
        else:           ff = flow.ff*fu(np.mod(time[k-1],tu))
            
            
        qtop[:] = numbaops.populate_wall_profile_target(flow.x,q,1.0)
        qlap_bc[:] = numbaops.laplacian_boundary_conditions_2D(flow.Re,flow.x,flow.y,q,qtop)
        qrhs[:] = qlap_bc - numbaops.evaluate_bilinearity_2D(flow.x,flow.y,q,q) + lops.L.dot(q) + ff

        if k == 0: 
            qfwd[:] = qrhs
            qkm1[:] = qrhs
        else:
            qfwd[:] = 1.5*qrhs - 0.5*qkm1
            qkm1[:] = qrhs
        
        qs[:] = lops.LL.dot(lops.LR.dot(q) + qfwd)
        p[:] = lops.luP.solve(lops.D.dot(qs))
        q[:] = qs - lops.G.dot(p)

        
        t1 = tlib.perf_counter() - t0
        ttot += t1
        # print("Time step: %d/%d,\t Iter time: %1.3e [s],\t Tot. time: %1.3e [s]"%(k,len(time),t1,ttot))
        
        if np.mod(k,n) == 0: 
            data[:,idx_save] = q
            idx_save += 1
            
            if np.isnan(np.linalg.norm(q)):
                raise ValueError ("Code blew up.")
            
            
    return data, tsave









