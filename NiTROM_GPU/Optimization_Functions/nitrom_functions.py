import numpy as np
import scipy as sp
import torch
import torch.distributed as dist
from string import ascii_lowercase as ascii
import pymanopt

import time as tlib
from ..PyTorch_Functions.integrators import my_etdrk4, etdrk4_setup
from ..PyTorch_Functions.linear_interpolation import Interp1D

def create_objective_and_gradient(manifold,opt_obj,pool,fom):
    
    """
    opt_obj:        instance of class "optimization_objects" in file "classes.py"
    pool:           instance of the class "pool" in file "classes.py"
    fom:            instance of the full-order model class
    """

    euclidean_hessian = None

    @pymanopt.function.pytorch(manifold)
    def cost(*params):

        """ 
            Evaluate the cost function 
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """
        
        Phi, Psi = params[0], params[1]
        tensors = params[2:]

        Phi = Phi.to(pool.device); Psi = Psi.to(pool.device)
        tensors = tuple(tensor.to(pool.device) for tensor in tensors)
        PhiF = Phi@torch.linalg.inv(Psi.T@Phi)

        D, V = torch.linalg.eig(tensors[0])
        V_inv = torch.linalg.inv(V)
        linop = V, D, V_inv
        dt = opt_obj.time[1] - opt_obj.time[0]
        etdrk4_coefs = etdrk4_setup(linop, dt)

        J = 0.0
        for k in range (pool.my_n_traj):
            
            # Integrate the reduced-order model from time t = 0 to the final time 
            # specified by the last snapshot in the training trajectory
            z0 = Psi.T@opt_obj.X[k,:,0]
            u = Psi.T@opt_obj.F[:,k]
            sol = my_etdrk4(etdrk4_coefs,opt_obj.evaluate_rom_rhs_nonlinear,opt_obj.time,z0,args=(u,)+tensors)
            e = fom.compute_output(opt_obj.X[k,:,:]) - fom.compute_output(PhiF@sol)
            J += (1./opt_obj.weights[k])*torch.trace(e.T@e)

        if opt_obj.l2_pen is not None and pool.rank == 0:
            time_pen = torch.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom,device=pool.device)
            Z = my_etdrk4(etdrk4_coefs,lambda t,z: 0*z,time_pen,opt_obj.randic)

            J += opt_obj.l2_pen*torch.dot(Z[:,-1],Z[:,-1])
        
        if pool.world_size > 1:
            dist.all_reduce(J, op=dist.ReduceOp.SUM)
        
        return J.cpu()
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):

        """ 
            Evaluate the euclidean gradient of the cost function with respect to the parameters
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """

        Phi, Psi = params[0], params[1]
        tensors = params[2:]

        Phi = torch.from_numpy(Phi).to(pool.device); Psi = torch.from_numpy(Psi).to(pool.device)
        tensors = tuple(torch.from_numpy(tensor).to(pool.device) for tensor in tensors)

        D, V = torch.linalg.eig(tensors[0])
        V_inv = torch.linalg.inv(V)
        linop = V, D, V_inv
        linop_T = V_inv.T, D, V.T
        dt = opt_obj.time[1] - opt_obj.time[0]
        dt2 = dt / (opt_obj.nsave_rom-1)
        etdrk4_coefs = etdrk4_setup(linop, dt)
        etdrk4_coefs_2 = etdrk4_setup(linop, dt2)
        etdrk4_coefs_T2 = etdrk4_setup(linop_T, dt2)
        t_unit = torch.linspace(0.0, 1.0, steps=opt_obj.nsave_rom, device=pool.device, dtype=torch.float64)
        
        # Initialize arrays to store the gradients
        n, r = Phi.shape
        grad_Phi = torch.zeros((n,r), device=pool.device)
        grad_Psi = torch.zeros((n,r), device=pool.device)
        grad_tensors = [torch.zeros_like(tensor) for tensor in tensors]
        

        # Initialize arrays needed for future computations
        lam_j_0 = torch.zeros(r, device=pool.device, dtype=Phi.dtype)
        Int_lambda = torch.zeros(r, device=pool.device)
        
        # Biorthogonalize Phi and Psi
        F = torch.linalg.inv(Psi.T@Phi)
        PhiF = Phi@F
        
        # Gauss-Legendre quadrature points and weights
        # Cubic spline interpolation to compute integral
        tlg, wlg = np.polynomial.legendre.leggauss(opt_obj.leggauss_deg)
        tlg = torch.from_numpy(tlg).to(pool.device); wlg = torch.from_numpy(wlg).to(pool.device)

        for k in range (pool.my_n_traj):

            z0 = Psi.T@opt_obj.X[k,:,0]
            u = Psi.T@opt_obj.F[:,k]
            sol = my_etdrk4(etdrk4_coefs,opt_obj.evaluate_rom_rhs_nonlinear,opt_obj.time,z0,args=(u,)+tensors)
            Z = sol
            e = fom.compute_output(opt_obj.X[k,:,:]) - fom.compute_output(PhiF@Z)
            alpha = opt_obj.weights[k]
            
            lam_j_0 *= 0.0
            Int_lambda *= 0.0
            
            for j in range (opt_obj.n_snapshots - 1):
                ej = e[:,opt_obj.n_snapshots - j - 1]
                zj = Z[:,opt_obj.n_snapshots - j - 1]
                Ctej = fom.compute_output_derivative(PhiF@zj).T@ej

                # Compute the sums in (2.13) and (2.14) in the arXiv paper. Notice that this loop sums backwards
                # from j = N-1 to j = 1, so we will compute the term j = 0 after this loop 
                grad_Psi += (2/alpha)*torch.einsum('i,j',PhiF@zj,PhiF.T@Ctej)
                grad_Phi += -(2/alpha)*torch.einsum('i,j',Ctej - Psi@(PhiF.T@Ctej),F@zj)


                # ------ Compute the fwd ROM solution between times t0_j and tf_j ---------
                id1 = opt_obj.n_snapshots - 1 - j
                id0 = id1 - 1
                
                tf_j = opt_obj.time[id1]
                t0_j = opt_obj.time[id0]
                z0_j = Z[:,id0]

                delta = tf_j - t0_j
                time_rom_j = t0_j + t_unit * delta
                if torch.abs(time_rom_j[-1] - tf_j) >= 1e-6:
                    print(time_rom_j[-1],tf_j)
                    raise ValueError("Error in euclidean_gradient() - final time is not correct!")

                sol_j = my_etdrk4(etdrk4_coefs_2,opt_obj.evaluate_rom_rhs_nonlinear,time_rom_j,z0_j,args=(u,)+tensors)
                Z_j = torch.fliplr(sol_j)
                fZ = Interp1D(time_rom_j,Z_j,extrapolate=True)
                # --------------------------------------------------------------------------

                # ------ Compute the adj ROM solution between times t0_j and tf_j ----------
                lam_j_0 += (2/alpha)*PhiF.T@Ctej
                sol_lam = my_etdrk4(etdrk4_coefs_T2,opt_obj.evaluate_rom_adjoint_nonlinear,time_rom_j,lam_j_0,args=(fZ,)+tensors)
                Lam = torch.fliplr(sol_lam)
                lam_j_0 = Lam[:,0]
                Z_j = torch.fliplr(Z_j)
                # --------------------------------------------------------------------------
                
                # Interpolate Z_j and Lam onto Gauss-Legendre points
                a = (tf_j - t0_j)/2
                b = (tf_j + t0_j)/2
                time_j_lg = a*tlg + b

                fZ = Interp1D(time_rom_j,Z_j,extrapolate=True)
                fL = Interp1D(time_rom_j,Lam,extrapolate=True)
                Z_j_lg = fZ(time_j_lg)
                Lam_lg = fL(time_j_lg)
                
                for i in range (opt_obj.leggauss_deg):
                    
                    
                    Int_lambda += a*wlg[i]*Lam_lg[:,i]
                    
                    for (count,p) in enumerate(opt_obj.poly_comp):
                        equation = ','.join(ascii[:p+1])
                        operands = [Lam_lg[:,i]] + [Z_j_lg[:,i] for _ in range (p)]
                        grad_tensors[count] -= a*wlg[i]*torch.einsum(equation,*operands)
                    
            
            # Add the term j = 0 in the sums (2.13) and (2.14). Also add the  
            # contribution of the initial condition (last term in (2.14)) to grad_Psi.
            # Add also the contribution of the steady forcing to grad_Psi.
            ej, zj = e[:,0], Z[:,0]
            Ctej = fom.compute_output_derivative(PhiF@zj).T@ej
            grad_Psi += (2/alpha)*torch.einsum('i,j',PhiF@zj,PhiF.T@Ctej) \
                        - torch.einsum('i,j',opt_obj.X[k,:,0],lam_j_0) \
                        - torch.einsum('i,j',opt_obj.F[:,k],Int_lambda)
            grad_Phi += -(2/alpha)*torch.einsum('i,j',Ctej - Psi@(PhiF.T@Ctej),F@zj)

        # Compute the gradient of the stability-promoting term
        if opt_obj.l2_pen is not None and pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor

            time_pen = torch.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom,device=pool.device)
            Z = my_etdrk4(etdrk4_coefs,lambda t,z: 0*z,time_pen,opt_obj.randic)
            Mu = my_etdrk4(etdrk4_coefs,lambda t,z: 0*z,time_pen,-2*opt_obj.l2_pen*Z[:,-1])
            Mu = torch.fliplr(Mu)
            
            for k in range (opt_obj.n_snapshots - 1):
                
                k0, k1 = k*opt_obj.nsave_rom, (k+1)*opt_obj.nsave_rom
                fZ = Interp1D(time_pen[k0:k1],Z[:,k0:k1],extrapolate=True)
                fMu = Interp1D(time_pen[k0:k1],Mu[:,k0:k1],extrapolate=True)
            
                a = (time_pen[k1-1] - time_pen[k0])/2
                b = (time_pen[k1-1] + time_pen[k0])/2
                time_k_lg = a*tlg + b
                
                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)
                
                for i in range (opt_obj.leggauss_deg):
                    grad_tensors[idx] += -a*wlg[i]*torch.einsum('i,j',Muk[:,i],Zk[:,i])

        if pool.world_size > 1:
            dist.all_reduce(grad_Phi, op=dist.ReduceOp.SUM)
            dist.all_reduce(grad_Psi, op=dist.ReduceOp.SUM)
            for k in range (len(grad_tensors)):
                dist.all_reduce(grad_tensors[k], op=dist.ReduceOp.SUM)

        if opt_obj.which_fix == 'fix_bases':

            grad_Phi *= 0.0; grad_Psi *= 0.0

        elif opt_obj.which_fix == 'fix_tensors':    
            
            for k in range (len(grad_tensors)): grad_tensors[k] *= 0.0

        grad_Phi = grad_Phi.cpu().numpy()
        grad_Psi = grad_Psi.cpu().numpy()
        grad_tensors = tuple(tensor.cpu().numpy() for tensor in grad_tensors)

        return grad_Phi, grad_Psi, *grad_tensors
    

    return cost, euclidean_gradient, euclidean_hessian


def check_gradient_using_finite_difference(M,Phi,Psi,A2,A3,A4,opt_obj,mpi_pool,fom,eps):

    cost, grad, _ = create_objective_and_gradient(M,opt_obj,mpi_pool,fom)
    gPhi, gPsi, gA2, gA3, gA4 = grad(Phi,Psi,A2,A3,A4)

    # Check Phi gradient 
    delta = sp.linalg.orth(np.random.randn(3,2))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi + eps*delta,Psi,A2,A3,A4) - cost(Phi - eps*delta,Psi,A2,A3,A4))
    dgrad = np.trace(delta.T@gPhi)
    error = np.abs(dfd - dgrad)
    percent_error = error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for Phi ------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check Psi gradient 
    delta = sp.linalg.orth(np.random.randn(3,2))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi + eps*delta,A2,A3,A4) - cost(Phi,Psi - eps*delta,A2,A3,A4))
    dgrad = np.trace(delta.T@gPsi)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for Psi ------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check A2 gradient 
    delta = np.random.randn(2,2)
    delta = delta/np.sqrt(np.trace(delta.T@delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi,A2 + eps*delta,A3,A4) - cost(Phi,Psi,A2 - eps*delta,A3,A4))
    dgrad = np.trace(delta.T@gA2)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A2 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")

    # Check A3 gradient 
    delta = np.random.randn(2,2,2)
    delta = delta/np.sqrt(np.einsum('ijk,ijk',delta,delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi,A2,A3 + eps*delta,A4) - cost(Phi,Psi,A2,A3 - eps*delta,A4))
    dgrad = np.einsum('ijk,ijk',delta,gA3)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A3 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check A4 gradient 
    delta = np.random.randn(2,2,2,2)
    delta = delta/np.sqrt(np.einsum('ijkl,ijkl',delta,delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)
        
    dfd = (0.5/eps)*(cost(Phi,Psi,A2,A3,A4 + eps*delta) - cost(Phi,Psi,A2,A3,A4 - eps*delta))
    dgrad = np.einsum('ijkl,ijkl',delta,gA4)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A4 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")