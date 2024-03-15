#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:34:37 2022

@author: alberto
"""

import numpy as np
import numba 




@numba.njit("f8[:](f8[:],f8[:])",cache=True) 
def populate_wall_profile(x,q):
    
    qbc = np.zeros(len(x)-1)
    for k in range (len(qbc)):
        qbc[k] = 2.0 - q[k]
        
    return qbc

# @numba.njit("f8[:](f8[:],f8[:],f8)",cache=True) 
def populate_wall_profile_target(x,q,target):
    
    qbc = np.zeros(len(x)-1)
    for k in range (len(qbc)):
        qbc[k] = 2*target - q[k]
        
    return qbc


@numba.njit("f8[:](f8,f8[:],f8[:],f8[:],f8[:])",cache=True) 
def laplacian_boundary_conditions_2D(Re,x,y,q,qbc):
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    rowsu = len(y)
    colsu = len(x) - 1
    szu = rowsu*colsu
    
    rowsv = len(y) - 1
    colsv = len(x) 
    szv = rowsv*colsv
    
    qout = np.zeros(szu+szv,dtype=np.float64)
    
    fx = 1./(dx*dx*Re)
    fy = 1./(dy*dy*Re)
    
    # --------------------------------------------------------
    # ---------- X momentum ----------------------------------
    # --------------------------------------------------------
    
    # bottom wall
    i = 0
    for j in range (colsu):
        ku = i*colsu + j
        qout[ku] = fy*qbc[j]
        
    # top wall
    i = rowsu-1
    for j in range (colsu):
        ku = i*colsu + j
        qout[ku] = fy*(-q[ku]) # (-) b/c we interpolate to 0
    

    # --------------------------------------------------------
    # ---------- Y momentum ----------------------------------
    # --------------------------------------------------------
    
    # right and left walls
    for j in [0,colsv-1]:
        for i in range (rowsv):
            kv = i*colsv + j + szu
            qout[kv] = fx*(-q[kv])  # (-) b/c we interpolate to 0
        
    
    return qout 


@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_bilinearity_2D(x,y,q1,q2):
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    rowsu = len(y)
    colsu = len(x) - 1
    szu = rowsu*colsu
    
    rowsv = len(y) - 1
    colsv = len(x) 
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- X momentum: x derivatives ----------
    # -----------------------------------------------
    for i in range (rowsu):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            duu_dx = (0.5*(q1[ku]+q1[ku+1])*0.5*(q2[ku]+q2[ku+1])  \
                    - 0.5*(q1[ku]+q1[ku-1])*0.5*(q2[ku]+q2[ku-1]))/dx
                
            qnl[ku] += duu_dx
    
    for j in [0,colsu-1]:
        if j == 0:  sgn = 1
        else:       sgn = -1
        
        for i in range (rowsu):
            
            duu_dx = sgn*(0.5*(q1[ku]+q1[ku+sgn])*0.5*(q2[ku]+q2[ku+sgn])  \
                    - 0.5*(q1[ku])*0.5*(q2[ku]))/dx
        
            qnl[ku] += duu_dx
            
    
    # -----------------------------------------------
    # ---------- X momentum: y derivatives ----------
    # -----------------------------------------------
    for i in range (1,rowsu-1):
        for j in range (colsu):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            dvu_dy = (0.5*(q1[kv]+q1[kv+1])*0.5*(q2[ku]+q2[ku+colsu]) \
                    -0.5*(q1[kv-colsv]+q1[kv-colsv+1])*0.5*(q2[ku]+q2[ku-colsu]))/dy
            
            qnl[ku] += dvu_dy
            

    for i in [0,rowsu-1]:
        if i == 0:  sgn = 1;    off = 0
        else:       sgn = -1;   off = -colsv
        
        for j in range (colsu):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            dvu_dy = sgn*(0.5*(q1[kv+off]+q1[kv+off+1])*0.5*(q2[ku]+q2[ku+sgn*colsu]))/dy 
            
            qnl[ku] += dvu_dy
            
    # -----------------------------------------------
    # ---------- Y momentum: x derivatives ----------
    # -----------------------------------------------
    for i in range (rowsv):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            duv_dx = (0.5*(q1[ku]+q1[ku+colsu])*0.5*(q2[kv]+q2[kv+1]) \
                    - 0.5*(q1[ku-1]+q1[ku-1+colsu])*0.5*(q2[kv]+q2[kv-1]))/dx
                
            qnl[kv] += duv_dx
                
    for j in [0,colsv-1]:
        if j == 0:  sgn = 1;    off = 0
        else:       sgn = -1;   off = -1
        
        for i in range (rowsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            duv_dx = sgn*(0.5*(q1[ku+off]+q1[ku+off+colsu])*0.5*(q2[kv]+q2[kv+sgn]))/dx 
            
            qnl[kv] += duv_dx
            
    
    # -----------------------------------------------
    # ---------- Y momentum: y derivatives ----------
    # -----------------------------------------------
    for i in range (1,rowsv-1):
        for j in range (colsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            dvv_dy = (0.5*(q1[kv]+q1[kv+colsv])*0.5*(q2[kv]+q2[kv+colsv]) \
                    - 0.5*(q1[kv]+q1[kv-colsv])*0.5*(q2[kv]+q2[kv-colsv]))/dy
                
            qnl[kv] += dvv_dy
                
    for i in [0,rowsv-1]:
        if i == 0:  sgn = 1
        else:       sgn = -1
        for j in range (colsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            dvv_dy = sgn*(0.5*(q1[kv]+q1[kv+sgn*colsv])*0.5*(q2[kv]+q2[kv+sgn*colsv]) \
                            - 0.5*q1[kv]*0.5*q2[kv])/dy
                    
            
            qnl[kv] += dvv_dy
    
    
    return qnl


@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_adjoint_bilinearity_2D(x,y,q1,q2):
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    rowsu = len(y)
    colsu = len(x) - 1
    szu = rowsu*colsu
    
    rowsv = len(y) - 1
    colsv = len(x) 
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- X momentum: x derivatives ----------
    # -----------------------------------------------
    for i in range (rowsu):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            qnl[ku] += (q1[ku+1]-q1[ku-1])*0.5*q2[ku]/dx
            qnl[ku+1] +=  (q1[ku]+q1[ku+1])*0.5*q2[ku]/dx
            qnl[ku-1] +=  -(q1[ku]+q1[ku-1])*0.5*q2[ku]/dx          
                           
                          
    
    for j in [0,colsu-1]: 
        if j == 0:  sgn = 1
        else:       sgn = -1
        
        for i in range (rowsu):
                
            qnl[ku] += sgn*q1[ku+sgn]*0.5*q2[ku]/dx
            qnl[ku+sgn] += sgn*(q1[ku]+q1[ku+sgn])*0.5*q2[ku]/dx
        
            
    
    # -----------------------------------------------
    # ---------- X momentum: y derivatives ----------
    # -----------------------------------------------
    for i in range (1,rowsu-1):
        for j in range (colsu):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            
            qnl[ku] += (0.5*(q1[kv]+q1[kv+1]) - 0.5*(q1[kv-colsv]+q1[kv-colsv+1]))*0.5*q2[ku]/dy
            qnl[ku+colsu] += 0.5*(q1[kv]+q1[kv+1])*0.5*q2[ku]/dy
            qnl[ku-colsu] += -0.5*(q1[kv-colsv]+q1[kv-colsv+1])*0.5*q2[ku]/dy
            
            qnl[kv] += 0.5*(q1[ku]+q1[ku+colsu])*0.5*q2[ku]/dy
            qnl[kv+1] += 0.5*(q1[ku]+q1[ku+colsu])*0.5*q2[ku]/dy
            qnl[kv-colsv] += -0.5*(q1[ku]+q1[ku-colsu])*0.5*q2[ku]/dy
            qnl[kv-colsv+1] += -0.5*(q1[ku]+q1[ku-colsu])*0.5*q2[ku]/dy
            
            

    for i in [0,rowsu-1]:
        if i == 0:  sgn = 1;    off = 0
        else:       sgn = -1;   off = -colsv
        
        for j in range (colsu):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu

            
            qnl[ku] += sgn*(0.5*(q1[kv+off]+q1[kv+off+1]))*0.5*q2[ku]/dy
            qnl[ku+sgn*colsu] += sgn*(0.5*(q1[kv+off]+q1[kv+off+1]))*0.5*q2[ku]/dy
            
            qnl[kv+off] += sgn*0.5*(q1[ku]+q1[ku+sgn*colsu])*0.5*q2[ku]/dy
            qnl[kv+off+1] += sgn*0.5*(q1[ku]+q1[ku+sgn*colsu])*0.5*q2[ku]/dy
            
    # -----------------------------------------------
    # ---------- Y momentum: x derivatives ----------
    # -----------------------------------------------
    for i in range (rowsv):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
                
            qnl[kv] += (0.5*(q1[ku]+q1[ku+colsu]) - 0.5*(q1[ku-1]+q1[ku-1+colsu]))*0.5*q2[kv]/dx
            qnl[kv+1] += 0.5*(q1[ku]+q1[ku+colsu])*0.5*q2[kv]/dx
            qnl[kv-1] += - 0.5*(q1[ku-1]+q1[ku-1+colsu])*0.5*q2[kv]/dx
            
            qnl[ku] += 0.5*(q1[kv]+q1[kv+1])*0.5*q2[kv]/dx
            qnl[ku+colsu] += 0.5*(q1[kv]+q1[kv+1])*0.5*q2[kv]/dx
            qnl[ku-1] += -0.5*(q1[kv]+q1[kv-1])*0.5*q2[kv]/dx
            qnl[ku-1+colsu] += -0.5*(q1[kv]+q1[kv-1])*0.5*q2[kv]/dx
                
    for j in [0,colsv-1]:
        if j == 0:  sgn = 1;    off = 0
        else:       sgn = -1;   off = -1
        
        for i in range (rowsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            qnl[kv] += sgn*0.5*(q1[ku+off]+q1[ku+off+colsu])*0.5*q2[kv]/dx
            qnl[kv+sgn] += sgn*0.5*(q1[ku+off]+q1[ku+off+colsu])*0.5*q2[kv]/dx
            
            qnl[ku+off] += sgn*0.5*(q1[kv]+q1[kv+sgn])*0.5*q2[kv]/dx
            qnl[ku+off+colsu] += sgn*0.5*(q1[kv]+q1[kv+sgn])*0.5*q2[kv]/dx
    
    # -----------------------------------------------
    # ---------- Y momentum: y derivatives ----------
    # -----------------------------------------------
    for i in range (1,rowsv-1):
        for j in range (colsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
                
            qnl[kv] += 2*(0.5*(q1[kv]+q1[kv+colsv]) - 0.5*(q1[kv]+q1[kv-colsv]))*0.5*q2[kv]/dy
            qnl[kv+colsv] += (q1[kv]+q1[kv+colsv])*0.5*q2[kv]/dy
            qnl[kv-colsv] += -(q1[kv]+q1[kv-colsv])*0.5*q2[kv]/dy
                
                
    for i in [0,rowsv-1]:
        if i == 0:  sgn = 1
        else:       sgn = -1
        for j in range (colsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            qnl[kv] += sgn*q1[kv+sgn*colsv]*0.5*q2[kv]/dy
            qnl[kv+sgn*colsv] += sgn*(q1[kv]+q1[kv+sgn*colsv])*0.5*q2[kv]/dy
    
    
    return qnl


