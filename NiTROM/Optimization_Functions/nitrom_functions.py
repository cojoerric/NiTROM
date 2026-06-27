import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from string import ascii_lowercase as ascii
import pymanopt
import time as tlib


from .utils import construct_operators, propagate_gradients

def rom_blowup_event(t, z, *args):
    return 1e4 - np.linalg.norm(z)
rom_blowup_event.terminal = True


def create_objective_and_gradient(*args, **kwargs):
    
    """
    opt_obj:        instance of class "optimization_objects" in file "nitrom_classes.py"
    mpi_pool:       instance of the class "mpi_pool" in file "nitrom_classes.py"
    fom:            instance of the full-order model class 
    """

    euclidean_hessian = None
    manifold, opt_obj, mpi_pool, fom = args
    poly_comp = opt_obj.poly_comp
    glob_stable = kwargs.get('glob_stable', False)

    @pymanopt.function.numpy(manifold)
    def cost(*params):

        """ 
            Evaluate the cost function 
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """
        
        
        Phi, Psi = params[0], params[1]
        tensors_old = params[2:]
        PhiF = Phi@sp.linalg.inv(Psi.T@Phi)
        if glob_stable:
            tensors, _ = construct_operators(tensors_old, poly_comp)
        else:
            tensors = tensors_old
        
        J = 0.0
        for k in range (opt_obj.my_n_traj): 

            # Integrate the reduced-order model from time t = 0 to the final time 
            # specified by the last snapshot in the training trajectory
            z0 = Psi.T@opt_obj.X[k,:,0]
            u = Psi.T@opt_obj.F[:,k]
            opt_obj.eval_counter = 0
            sol = solve_ivp(opt_obj.evaluate_rom_rhs,[0,opt_obj.time[-1]],z0,\
                            method='RK45',t_eval=opt_obj.time,args=(u,) + tensors, events=rom_blowup_event)
            y = sol.y
            if y.shape[1] < len(opt_obj.time):
                padding = np.repeat(y[:, -1:], len(opt_obj.time) - y.shape[1], axis=1)
                y = np.hstack((y, padding))
            e = fom.compute_output(opt_obj.X[k,:,:]) - fom.compute_output(PhiF@y)
            J += (1./opt_obj.weights[k])*np.trace(e.T@e)
        
        if opt_obj.l2_pen != None and mpi_pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
            time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t,z: tensors[idx]@z if np.linalg.norm(z) < 1e4 else 0*z,\
                           [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen)).y
            
            J += opt_obj.l2_pen*np.dot(Z[:,-1],Z[:,-1])
            
        J = np.sum(np.asarray(mpi_pool.comm.allgather(J)))
        J = mpi_pool.comm.bcast(J, root=0)
        
        return J

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params): 

        """ 
            Evaluate the euclidean gradient of the cost function with respect to the parameters
            Phi and Psi:    bases (size N x r) that define the projection operator
            tensors:        (A2,A3,...)
        """

        Phi, Psi = params[0], params[1]
        tensors_old = params[2:]
        if glob_stable:
            tensors, other_tensors = construct_operators(tensors_old, poly_comp)
        else:
            tensors = tensors_old
        
        # Initialize arrays to store the gradients
        n, r = Phi.shape
        grad_Phi = np.zeros((n,r))
        grad_Psi = np.zeros((n,r))
        grad_tensors = [0]*len(tensors)
        

        # Initialize arrays needed for future computations
        lam_j_0 = np.zeros(r)
        Int_lambda = np.zeros(r)
        
        # Biorthogonalize Phi and Psi
        F = sp.linalg.inv(Psi.T@Phi)
        PhiF = Phi@F
        
        # Gauss-Legendre quadrature points and weights
        tlg, wlg = np.polynomial.legendre.leggauss(opt_obj.leggauss_deg)
        wlg = np.asarray(wlg)
        
        import time as tlib
        for k in range (opt_obj.my_n_traj):

            z0 = Psi.T@opt_obj.X[k,:,0]
            u = Psi.T@opt_obj.F[:,k]
            opt_obj.eval_counter = 0
            sol = solve_ivp(opt_obj.evaluate_rom_rhs,[0,opt_obj.time[-1]],z0,\
                            method='RK45',t_eval=opt_obj.time,args=(u,) + tensors, events=rom_blowup_event)
            Z = sol.y
            if Z.shape[1] < len(opt_obj.time):
                padding = np.repeat(Z[:, -1:], len(opt_obj.time) - Z.shape[1], axis=1)
                Z = np.hstack((Z, padding))
            e = fom.compute_output(opt_obj.X[k,:,:]) - fom.compute_output(PhiF@Z)
            alpha = opt_obj.weights[k]
            
            lam_j_0 *= 0.0
            Int_lambda *= 0.0
            
            sum_sol_j_time = 0.0
            sum_sol_j_nfev = 0
            sum_sol_lam_time = 0.0
            sum_sol_lam_nfev = 0

            print(f"[Rank {mpi_pool.rank}] grad snapshot loops starting for traj {k}...", flush=True)
            for j in range (opt_obj.n_snapshots - 1):
        
                ej = e[:,opt_obj.n_snapshots - j - 1]
                zj = Z[:,opt_obj.n_snapshots - j - 1]
                Ctej = fom.compute_output_derivative(PhiF@zj).T@ej

                # Compute the sums in (2.13) and (2.14) in the arXiv paper. Notice that this loop sums backwards
                # from j = N-1 to j = 1, so we will compute the term j = 0 after this loop 
                grad_Psi += (2/alpha)*np.einsum('i,j',PhiF@zj,PhiF.T@Ctej)
                grad_Phi += -(2/alpha)*np.einsum('i,j',Ctej - Psi@(PhiF.T@Ctej),F@zj)


                # ------ Compute the fwd ROM solution between times t0_j and tf_j ---------
                id1 = opt_obj.n_snapshots - 1 - j
                id0 = id1 - 1
                
                tf_j = opt_obj.time[id1]
                t0_j = opt_obj.time[id0]
                z0_j = Z[:,id0]
                
                time_rom_j = np.linspace(t0_j,tf_j,num=opt_obj.nsave_rom,endpoint=True)
                if np.abs(time_rom_j[-1] - tf_j) >= 1e-10:
                    raise ValueError("Error in euclidean_gradient() - final time is not correct!")
                
                opt_obj.eval_counter = 0
                sol_j = solve_ivp(opt_obj.evaluate_rom_rhs,[t0_j,tf_j],z0_j,method='RK45',\
                                  t_eval=time_rom_j,args=(u,) + tensors, events=rom_blowup_event)
                Z_j = sol_j.y
                if Z_j.shape[1] < len(time_rom_j):
                    padding = np.repeat(Z_j[:, -1:], len(time_rom_j) - Z_j.shape[1], axis=1)
                    Z_j = np.hstack((Z_j, padding))
                Z_j = np.fliplr(Z_j)
                fZ = sp.interpolate.interp1d(time_rom_j,Z_j,kind='linear',fill_value='extrapolate')
                # --------------------------------------------------------------------------

                # ------ Compute the adj ROM solution between times t0_j and tf_j ----------
                lam_j_0 += (2/alpha)*PhiF.T@Ctej
                opt_obj.adj_eval_counter = 0
                sol_lam = solve_ivp(opt_obj.evaluate_rom_adjoint,[t0_j,tf_j],lam_j_0,\
                                    method='RK45',t_eval=time_rom_j,args=(fZ,) + tensors, events=rom_blowup_event)
                Lam = sol_lam.y
                if Lam.shape[1] < len(time_rom_j):
                    padding = np.repeat(Lam[:, -1:], len(time_rom_j) - Lam.shape[1], axis=1)
                    Lam = np.hstack((Lam, padding))
                Lam = np.fliplr(Lam)
                lam_j_0 = Lam[:,0]
                Z_j = np.fliplr(Z_j)
                # --------------------------------------------------------------------------
                
                # Interpolate Z_j and Lam onto Gauss-Legendre points
                a = (tf_j - t0_j)/2
                b = (tf_j + t0_j)/2
                time_j_lg = a*tlg + b

                fZ = sp.interpolate.interp1d(time_rom_j,Z_j,kind='linear',fill_value='extrapolate')
                fL = sp.interpolate.interp1d(time_rom_j,Lam,kind='linear',fill_value='extrapolate')
                Z_j_lg = fZ(time_j_lg)
                Lam_lg = fL(time_j_lg)
                
                for i in range (opt_obj.leggauss_deg):
                    
                    Int_lambda += a*wlg[i]*Lam_lg[:,i]
                    
                    for (count,p) in enumerate(opt_obj.poly_comp):
                        equation = ','.join(ascii[:p+1])
                        operands = [Lam_lg[:,i]] + [Z_j_lg[:,i] for _ in range (p)]
                        grad_tensors[count] -= a*wlg[i]*np.einsum(equation,*operands)
                    
            
            # Add the term j = 0 in the sums (2.13) and (2.14). Also add the  
            # contribution of the initial condition (last term in (2.14)) to grad_Psi.
            # Add also the contribution of the steady forcing to grad_Psi.
            ej, zj = e[:,0], Z[:,0]
            Ctej = fom.compute_output_derivative(PhiF@zj).T@ej
            grad_Psi += (2/alpha)*np.einsum('i,j',PhiF@zj,PhiF.T@Ctej) \
                        - np.einsum('i,j',opt_obj.X[k,:,0],lam_j_0) \
                        - np.einsum('i,j',opt_obj.F[:,k],Int_lambda)
            grad_Phi += -(2/alpha)*np.einsum('i,j',Ctej - Psi@(PhiF.T@Ctej),F@zj)

        
        # Compute the gradient of the stability-promoting term
        if opt_obj.l2_pen != None and mpi_pool.rank == 0:
            
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
            
            A = tensors[idx]
            
            time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
            opt_obj.eval_counter = 0
            sol_Z = solve_ivp(lambda t,z: A@z,\
                           [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen, events=rom_blowup_event)
            Z = sol_Z.y
            if Z.shape[1] < len(time_pen):
                padding = np.repeat(Z[:, -1:], len(time_pen) - Z.shape[1], axis=1)
                Z = np.hstack((Z, padding))
            opt_obj.adj_eval_counter = 0
            sol_Mu = solve_ivp(lambda t,z: A.T@z,\
                           [0,time_pen[-1]],-2*opt_obj.l2_pen*Z[:,-1],method='RK45',t_eval=time_pen, events=rom_blowup_event)
            Mu = sol_Mu.y
            if Mu.shape[1] < len(time_pen):
                padding = np.repeat(Mu[:, -1:], len(time_pen) - Mu.shape[1], axis=1)
                Mu = np.hstack((Mu, padding))
            Mu = np.fliplr(Mu)
            
            
            for k in range (opt_obj.n_snapshots - 1):
                
                k0, k1 = k*opt_obj.nsave_rom, (k+1)*opt_obj.nsave_rom
                fZ = sp.interpolate.interp1d(time_pen[k0:k1],Z[:,k0:k1],kind='linear',fill_value='extrapolate')
                fMu = sp.interpolate.interp1d(time_pen[k0:k1],Mu[:,k0:k1],kind='linear',fill_value='extrapolate')
            
                a = (time_pen[k1-1] - time_pen[k0])/2
                b = (time_pen[k1-1] + time_pen[k0])/2
                time_k_lg = a*tlg + b
                
                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)
                
                for i in range (opt_obj.leggauss_deg):
                    grad_tensors[idx] += -a*wlg[i]*np.einsum('i,j',Muk[:,i],Zk[:,i])
                

        if opt_obj.which_fix == 'fix_bases':

            grad_Phi *= 0.0; grad_Psi *= 0.0
            for k in range (len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))


        elif opt_obj.which_fix == 'fix_tensors':    
            
            for k in range (len(grad_tensors)): grad_tensors[k] *= 0.0
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))


        else: 

            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))
            for k in range (len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))

        grad_Phi = mpi_pool.comm.bcast(grad_Phi, root=0)
        grad_Psi = mpi_pool.comm.bcast(grad_Psi, root=0)
        for k in range (len(grad_tensors)):
            grad_tensors[k] = mpi_pool.comm.bcast(grad_tensors[k], root=0)

        if glob_stable:
            grad_tensors_new = propagate_gradients(grad_tensors, other_tensors, tensors_old, poly_comp)
        else:
            grad_tensors_new = grad_tensors

        return grad_Phi, grad_Psi, *grad_tensors_new
    

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





