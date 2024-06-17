import numpy as np 
import scipy as sp
import scipy.linalg as sciplin
from scipy.integrate import solve_ivp
import pymanopt


def create_objective_and_gradient(manifold,opt_obj,mpi_pool,fom):

    euclidean_hessian = None

    @pymanopt.function.numpy(manifold)
    def cost(Phi,Psi):
        
        PhiF = Phi@sciplin.inv(Psi.T@Phi)               # Scaled Phi modes
        tensors, _ = fom.assemble_petrov_galerkin_tensors(PhiF,Psi)
        
        J = 0.0
        for k in range (mpi_pool.my_n_traj): 

            # Integrate the reduced-order model from time t = 0 to the final time 
            # specified by the last snapshot in the training trajectory
            z0 = Psi.T@opt_obj.X[k,:,0]
            u = Psi.T@opt_obj.F[:,k]
            sol = solve_ivp(opt_obj.evaluate_rom_rhs,[0,opt_obj.time[-1]],z0,\
                            method='RK45',t_eval=opt_obj.time,args=(u,) + tensors)

            # Compute the error
            e = fom.compute_output(opt_obj.X[k,:,:]) - fom.compute_output(PhiF@sol.y)
            J += (1./opt_obj.weights[k])*np.trace(e.T@e)
        
        return np.sum(np.asarray(mpi_pool.comm.allgather(J)))

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(Phi,Psi): 
            
        # Initialize arrays for gradients
        n, r = Phi.shape
        
        grad_Phi = np.zeros((n,r))
        grad_Psi = np.zeros((n,r))
        
        # Biorthogonalize Phi and Psi and assemble ROM
        F = sciplin.inv(Psi.T@Phi)
        PhiF = Phi@F
        tensors, _ = fom.assemble_petrov_galerkin_tensors(PhiF,Psi)
        
        # Gauss-Legendre quadrature points and weights
        tlg, wlg = np.polynomial.legendre.leggauss(opt_obj.leggauss_deg)
        wlg = np.asarray(wlg)
        
        for k in range (mpi_pool.my_n_traj):

            
            z0 = Psi.T@opt_obj.X[k,:,0]
            u = Psi.T@opt_obj.F[:,k]
            sol = solve_ivp(opt_obj.evaluate_rom_rhs,[0,opt_obj.time[-1]],z0,\
                            method='RK45',t_eval=opt_obj.time,args=(u,) + tensors)

            Z = sol.y
            e = fom.compute_output(opt_obj.X[k,:,:]) - fom.compute_output(PhiF@Z)
            alpha = opt_obj.weights[k]                
            
            # Initialize arrays needed for future computations
            lam_j_0 = np.zeros(r)
            Int_Psi = np.zeros((n,r))
            Int_Phi = np.zeros((n,r))
            
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
                    print(time_rom_j[-1],tf_j)
                    raise ValueError("Error in euclidean_gradient() - final time is not correct!")
                
                sol_j = solve_ivp(opt_obj.evaluate_rom_rhs,[t0_j,tf_j],z0_j,method='RK45',\
                                      t_eval=time_rom_j,args=(u,) + tensors)
                Z_j = np.fliplr(sol_j.y)
                fZ = sp.interpolate.interp1d(time_rom_j,Z_j,kind='linear',fill_value='extrapolate')
                # --------------------------------------------------------------------------

                # ------ Compute the adj ROM solution between times t0_j and tf_j ----------
                lam_j_0 += (2/alpha)*PhiF.T@Ctej
                sol_lam = solve_ivp(opt_obj.evaluate_rom_adjoint,[t0_j,tf_j],lam_j_0,\
                                    method='RK45',t_eval=time_rom_j,args=(fZ,) + tensors)
                Lam = np.fliplr(sol_lam.y)
                lam_j_0 = Lam[:,0]
                Z_j = np.fliplr(Z_j)
                # --------------------------------------------------------------------------
                
                # Interpolate Z_j and Lam onto Gauss-Legendre points
                a = (tf_j - t0_j)/2
                b = (tf_j + t0_j)/2
                time_j_lg = a*tlg + b

                fQ = sp.interpolate.interp1d(time_rom_j,PhiF@Z_j,kind='linear',fill_value='extrapolate')
                fL = sp.interpolate.interp1d(time_rom_j,Lam,kind='linear',fill_value='extrapolate')
 
                Q_j_lg = fQ(time_j_lg)
                Lam_lg = fL(time_j_lg)
                
                for i in range (opt_obj.leggauss_deg):
                    
                    ti = time_j_lg[i]
                    qi = Q_j_lg[:,i]
                    li = Lam_lg[:,i]

                    # Compute integral contribution to grad_Psi
                    term_1_psi = fom.evaluate_fom_dynamics(ti,qi,opt_obj.F[:,k])
                    term_2_psi = PhiF.T@fom.evaluate_fom_adjoint(ti,Psi@li,fQ)
                    Int_Psi += a*wlg[i]*(np.einsum('i,j',term_1_psi,li) - np.einsum('i,j',qi,term_2_psi))
                    

                    # Compute integral contribution to grad_Phi
                    term_1_phi = fom.evaluate_fom_adjoint(ti,Psi@li,fQ)
                    term_1_phi -= Psi@(PhiF.T@term_1_phi)
                    Int_Phi += a*wlg[i]*np.einsum('i,j',term_1_phi,F@(Psi.T@qi))
            
            grad_Psi += -Int_Psi 
            grad_Phi += -Int_Phi
            
            # Add the term j = 0 in the sums (2.13) and (2.14). Also add the  
            # contribution of the initial condition (last term in (2.14)) to grad_Psi.
            ej = e[:,0]
            zj = Z[:,0]
            Ctej = fom.compute_output_derivative(PhiF@zj).T@ej
            grad_Psi += (2/alpha)*np.einsum('i,j',PhiF@zj,PhiF.T@Ctej) - np.einsum('i,j',opt_obj.X[k,:,0],lam_j_0)
            grad_Phi += -(2/alpha)*np.einsum('i,j',Ctej - Psi@(PhiF.T@Ctej),F@zj)
            
        
        grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
        grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))

        
        return grad_Phi, grad_Psi

    return cost, euclidean_gradient, euclidean_hessian
    
