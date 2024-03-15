#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:38:17 2022

@author: alberto
"""
import numpy as np
import scipy.sparse
import linear_operators as lops
from scipy.sparse.linalg import splu
import scipy.linalg as sciplin
import numba_operators as numbaops


class flow_parameters:
    
    def __init__(self,Lx,Ly,Nx,Ny,Re):
        
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny
        self.x = self.dx*np.arange(0,self.Nx,1) + self.dx/2 - 0
        self.y = self.dy*np.arange(0,self.Ny,1) + self.dy/2 - 0
        
        self.Re = Re
        
        self.rowsu = self.Ny
        self.colsu = self.Nx - 1
        self.rowsv = self.Ny - 1
        self.colsv = self.Nx 
        self.rowsp = self.Ny
        self.colsp = self.Nx
        
        self.szu = self.colsu*self.rowsu
        self.szv = self.colsv*self.rowsv
        self.szp = self.colsp*self.rowsp
        
        self.q_sbf = np.zeros(self.szu+self.szv)
        
        
class linear_operators_2D:
    
    def __init__(self,flow,dt):
        
        self.L = lops.laplacian_2D(flow)
        self.I = scipy.sparse.identity(self.L.shape[0])
        self.LR = (1/dt)*self.I 
        self.LL = dt*self.I
        self.G = lops.gradient_2D(flow)
        self.D = self.G.T
        self.luP = splu((self.D.dot((self.G)).tocsc()))
        

class fom_class:
    
    def __init__(self,flow_params,lops):
        
        self.flow_params = flow_params
        self.lops = lops
        
    
    
    def compute_output(self,q):
        return q
    
    def compute_output_derivative(self,q):
        return scipy.sparse.identity(len(q))
    
    
    def assemble_forcing_profile(self,xc,yc): 
        
        dx = self.flow_params.x[1] - self.flow_params.x[0] 
        f = np.zeros(self.flow_params.szu + self.flow_params.szv)
        
        for i in range (self.flow_params.rowsu): 
            
            yi = self.flow_params.y[i]
            
            for j in range (self.flow_params.colsu): 
                
                xj = self.flow_params.x[j] + dx/2 
                f[i*self.flow_params.colsu+j] = np.exp(-5000*((xj - xc)**2 + (yi - yc)**2))
                
        f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f))) 
        self.f = f/np.linalg.norm(f)
    
    
    def assemble_petrov_galerkin_tensors(self,Phi,Psi): 

        x = self.flow_params.x
        y = self.flow_params.y
        Re = self.flow_params.Re
        q_sbf = self.flow_params.q_sbf
        
        PhiF = Phi@sciplin.inv(Psi.T@Phi) 
        r = Phi.shape[-1]

        A2 = np.zeros((r,r))
        A3 = np.zeros((r,r,r))

        for k in range (r): 
            vec = PhiF[:,k]

            qtop = numbaops.populate_wall_profile_target(x,vec,0.0) 
            qlap_bc = numbaops.laplacian_boundary_conditions_2D(Re,x,y,vec,qtop) 
            f = qlap_bc + self.lops.L.dot(vec) - numbaops.evaluate_bilinearity_2D(x,y,q_sbf,vec) -\
                numbaops.evaluate_bilinearity_2D(x,y,vec,q_sbf)
            f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f))) 
            A2[:,k] = Psi.T@f 

        for j in range (r): 
            vecj = PhiF[:,j] 
            
            for k in range (r): 
                veck = PhiF[:,k] 

                f = -numbaops.evaluate_bilinearity_2D(x,y,vecj,veck)
                f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f))) 
                A3[:,j,k] = Psi.T@f 


        return (A2, A3), (Psi.T@self.f,)
    
    
    def evaluate_fom_dynamics(self,vec): 
        
        x = self.flow_params.x
        y = self.flow_params.y
        Re = self.flow_params.Re
        q_sbf = self.flow_params.q_sbf
        
        qtop = numbaops.populate_wall_profile_target(x,vec,0.0) 
        qlap_bc = numbaops.laplacian_boundary_conditions_2D(Re,x,y,vec,qtop) 
        f = qlap_bc + self.lops.L.dot(vec) - numbaops.evaluate_bilinearity_2D(x,y,q_sbf,vec) -\
            numbaops.evaluate_bilinearity_2D(x,y,vec,q_sbf) - \
            numbaops.evaluate_bilinearity_2D(x,y,vec,vec)
        f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f))) 
        
        return f
    
    
    def evaluate_fom_fullrhs(self,q):
        
        x = self.flow_params.x
        y = self.flow_params.y
        Re = self.flow_params.Re
        vec = self.flow_params.q_sbf + q
        
        qtop = numbaops.populate_wall_profile_target(x,vec,1.0) 
        qlap_bc = numbaops.laplacian_boundary_conditions_2D(Re,x,y,vec,qtop) 
        f = qlap_bc + self.lops.L.dot(vec) - numbaops.evaluate_bilinearity_2D(x,y,vec,vec)
        f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f))) 
        
        return f
        
        
        
        
        
        
        
        
        
        
        
        