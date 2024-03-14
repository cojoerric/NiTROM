import numpy as np
import scipy
import scipy.linalg as sciplin

class CGL: 
    
    def __init__(self,x,nu,gamma,mu0,mu2,a): 
        
        # Load grid
        self.x = x 
        self.nx = len(x) 
        self.dx = x[1] - x[0]
        
        # Load parameters
        self.nu = nu 
        self.gamma = gamma 
        self.mu0 = mu0 
        self.mu2 = mu2 
        self.a = a
        self.cu = self.nu.imag/2
        self.cd = self.gamma.imag
        self.mu = (self.mu0 - self.cu**2) + self.mu2*(self.x**2)/2
        
        
        # Define derivative operators  
        self.assemble_first_derivative_operator()
        self.assemble_second_derivative_operator()
        self.muI = scipy.sparse.diags(self.mu,offsets=0,format='csc')
        self.assemble_linear_operator()
        
        # Define output & input operator
        self.xb = np.sqrt(-2*(self.mu0 - self.cu**2)/self.mu2)
        C = np.exp(-((self.x - self.xb)/1.6)**2)
        Cr = np.concatenate((C,np.zeros(self.nx))).reshape((1,-1))
        Ci = np.concatenate((np.zeros(self.nx),C)).reshape((1,-1))
        self.C = np.concatenate((Cr,Ci),axis=0)
        
        B = np.exp(-((self.x + self.xb)/1.6)**2)
        Br = np.concatenate((B,np.zeros(self.nx))).reshape((-1,1))
        Bi = np.concatenate((np.zeros(self.nx),B)).reshape((-1,1))
        self.B = np.concatenate((Br,Bi),axis=1)
        
        
        
    def assemble_first_derivative_operator(self): 
        
        f = 1./(12*self.dx)
        s = 1./(2*self.dx)
        
        rows = []
        cols = []
        data = [] 
        
        for i in range (2,self.nx-2):
            
            rows.extend([i,i,i,i])
            cols.extend([i-2,i-1,i+1,i+2])
            data.extend([f,-8*f,8*f,-f])
        
        i = 1
        rows.extend([i,i,i])
        cols.extend([i-1,i+1,i+2])
        data.extend([-8*f,8*f,-f]) 
        
        i = self.nx - 2 
        rows.extend([i,i,i])
        cols.extend([i-2,i-1,i+1]) 
        data.extend([f,-8*f,8*f]) 
        
        i = 0 
        rows.append(i)
        cols.append(i+1)
        data.append(s) 
        
        i = self.nx - 1 
        rows.append(i)
        cols.append(i-1) 
        data.append(-s)
        
        self.D = scipy.sparse.csc_array((data,(rows,cols)),shape=(self.nx,self.nx))
        
    
    def assemble_second_derivative_operator(self): 
        
        f = 1./(12*(self.dx**2))
        s = 1./(self.dx**2) 
        
        rows = []
        cols = []
        data = [] 
        
        for i in range (2,self.nx-2):
            
            rows.extend([i,i,i,i,i])
            cols.extend([i-2,i-1,i,i+1,i+2])
            data.extend([-f,16*f,-30*f,16*f,-f])
            
            
        
        i = 1
        rows.extend([i,i,i,i])
        cols.extend([i-1,i,i+1,i+2])
        data.extend([16*f,-30*f,16*f,-f])
        
        i = self.nx - 2 
        rows.extend([i,i,i,i])
        cols.extend([i-2,i-1,i,i+1])
        data.extend([-f,16*f,-30*f,16*f]) 
        
        i = 0 
        rows.extend([i,i])
        cols.extend([i,i+1]) 
        data.extend([-2*s,s]) 
        
        i = self.nx - 1
        rows.extend([i,i])
        cols.extend([i-1,i]) 
        data.extend([s,-2*s]) 
        
        self.DD = scipy.sparse.csc_array((data,(rows,cols)),shape=(self.nx,self.nx))
        

    def assemble_linear_operator(self):
        
        A11 = -self.nu.real*self.D + self.gamma.real*self.DD + self.muI 
        A12 = self.nu.imag*self.D - self.gamma.imag*self.DD 
        A21 = - A12 
        A22 = A11 
        
        self.A = scipy.sparse.bmat([[A11,A12],[A21,A22]],format='csc')
        

    def evaluate_cgl_nonlinearity(self,q1,q2,q3):
        
        nx = self.nx
        
        q1q2_real = q1[:nx]*q2[:nx] + q1[nx:]*q2[nx:] 
        
        q1q2q3_real = q1q2_real*q3[:nx] 
        q1q2q3_imag = q1q2_real*q3[nx:] 
        
        return -self.a*np.concatenate((q1q2q3_real,q1q2q3_imag))
    
    
    def evaluate_right_hand_side(self,q):
        return self.A.dot(q) + self.evaluate_cgl_nonlinearity(q,q,q)
    
    
    def evaluate_cgl_linearized_nonlinearity(self,Q,q):
        
        nx = self.nx 
        
        qreal = (3*Q[:nx]**2 + Q[nx:]**2)*q[:nx] + 2*Q[:nx]*Q[nx:]*q[nx:]
        qimag = 2*Q[:nx]*Q[nx:]*q[:nx] + (3*Q[nx:]**2 + Q[:nx]**2)*q[nx:]
        
        return -self.a*np.concatenate((qreal,qimag))
    
    def evaluate_cgl_adjoint_rhs(self,Q,q):
        return self.A.T.dot(q) + self.evaluate_cgl_linearized_nonlinearity(Q,q)
    
    
    def assemble_random_periodic_vector(self,freqs,time):
        
        nf = len(freqs)
        modes = np.random.randn(2*self.nx,nf) + 1j*np.random.randn(2*self.nx,nf)
        modes[:,0] = modes[:,0].real
        
        # Normalize the modes so that the forcing is unit norm
        val = np.dot(modes[:,0],modes[:,0])
        for k in range (1,nf):  val += 2*np.dot(modes[:,k].conj(),modes[:,k])
        modes = np.sqrt(1/val)*modes
        
        f = np.outer(modes[:,0],np.ones(len(time))).real
        for k in range (1,len(freqs)):  f += 2*np.outer(modes[:,k],np.exp(1j*freqs[k]*time)).real
            
        return f.real
    
    
    """
        Functions that conform with those expected by the TrOOP during the 
        optimization
    """
    
    def evaluate_fom_dynamics(self,t,q,u):
        
        f = u.copy() if hasattr(u,"__len__") == True else u(t)
        if np.linalg.norm(q) >= 1e4:    vec = 0*q
        else:                           vec = self.evaluate_right_hand_side(q) + f
        
        return vec
    
    def evaluate_fom_adjoint(self,t,q,fQ):
        
        if np.linalg.norm(q) >= 1e4:    vec = 0*q
        else:                           vec = self.evaluate_cgl_adjoint_rhs(fQ(t),q)
        
        return vec
    
    def compute_output(self,q):
        return self.C@q
    
    def compute_output_derivative(self,q):
        return self.C
    
    def assemble_petrov_galerkin_tensors(self,Phi,Psi):
        
        n, r = Phi.shape
        PhiF = Phi@sciplin.inv(Psi.T@Phi)
        
        Br = Psi.T@self.B
        Cr = self.C@PhiF
        Ar = Psi.T@(self.A.dot(PhiF))
        Hr = np.zeros((r,r,r,r))
        
        for i in range (r):
            for j in range (r):
                for k in range (r): 
                    Hr[:,i,j,k] = Psi.T@(self.evaluate_cgl_nonlinearity(PhiF[:,i],PhiF[:,j],PhiF[:,k]))
        
        return (Ar, Hr), (Br, Cr)
    

    
class time_step_cgl: 
    
    def __init__(self,cgl,time): 
        
        self.time = time
        self.dt = time[1] - time[0]
        self.Id = scipy.sparse.identity(2*cgl.nx,format='csc')
        self.lu_A = scipy.sparse.linalg.splu((1/self.dt)*self.Id - (1/2)*cgl.A)
        self.lu_AT = scipy.sparse.linalg.splu((1/self.dt)*self.Id - (1/2)*cgl.A.T)
            
            
    
    def time_step(self,cgl,q,nsave,*argv): 
        
        tsave = self.time[::nsave]
        Q = np.zeros((2*cgl.nx,len(tsave)))
        Q[:,0] = q
        
        if len(argv) == 0:
            k_save = 1
            for k in range (1,len(self.time)):
                
                # print("CGL solver. Timestep %d/%d"%(k,len(self.time)))
                qnl_k = cgl.evaluate_cgl_nonlinearity(q,q,q)
                
                if k == 1: 
                    qnl_km1 = qnl_k
                    qnl_fwd = qnl_k
                else:
                    qnl_k = cgl.evaluate_cgl_nonlinearity(q,q,q)
                    qnl_fwd = (3./2)*qnl_k - (1./2)*qnl_km1 
                    qnl_km1 = qnl_k 
                
                q = self.lu_A.solve((1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd)
                
                if np.mod(k,nsave) == 0: 
                    Q[:,k_save] = q
                    k_save += 1
        else:
            
            fU = argv[0] 
            TfU = argv[1]
            k_save = 1
            for k in range (1,len(self.time)):
                
                # print("CGL solver. Timestep %d/%d"%(k,len(self.time)))
                qnl_k = cgl.evaluate_cgl_nonlinearity(q,q,q)
                
                if k == 1: 
                    qnl_km1 = qnl_k
                    qnl_fwd = qnl_k
                else:
                    qnl_k = cgl.evaluate_cgl_nonlinearity(q,q,q)
                    qnl_fwd = (3./2)*qnl_k - (1./2)*qnl_km1 
                    qnl_km1 = qnl_k 
                
                q = self.lu_A.solve((1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd + fU(np.mod(self.time[k-1],TfU)))
                
                if np.mod(k,nsave) == 0: 
                    Q[:,k_save] = q
                    k_save += 1
            
                    
        return Q, cgl.C@Q, tsave
    
    
    def time_step_linearized(self,cgl,Qbflow,tbflow,q,*argv):
        
        fQ = scipy.interpolate.interp1d(tbflow,Qbflow,kind='linear',fill_value='extrapolate')
        Tb = tbflow[-1] + (tbflow[1] - tbflow[0])
        
        forcing_flag = 0
        if len(argv) > 0: 
            fF = argv[0]
            Tf = argv[1]
            forcing_flag = 1
        
        for k in range (1,len(self.time)):
            
            tk = self.time[k] - 1
            qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(np.mod(tk,Tb)),q)
            
            if k == 1: 
                qnl_km1 = qnl_k
                qnl_fwd = qnl_k
            else:
                qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(np.mod(tk,Tb)),q)
                qnl_fwd = (3./2)*qnl_k - (1./2)*qnl_km1 
                qnl_km1 = qnl_k 
                
            vec_rhs = (1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd
            if forcing_flag == 1:
                vec_rhs = (1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd + fF(np.mod(tk,Tf))
                
            q = self.lu_A.solve(vec_rhs)
            
                
        return q
    
    
    def time_step_adjoint(self,cgl,Qbflow,tbflow,q):
        
        fQ = scipy.interpolate.interp1d(tbflow,np.fliplr(Qbflow),kind='linear',fill_value='extrapolate')
        Tb = tbflow[-1] + (tbflow[1] - tbflow[0])

        for k in range (1,len(self.time)):
            
            tk = self.time[k] - 1
            qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(np.mod(tk,Tb)),q)
            
            if k == 1: 
                qnl_km1 = qnl_k
                qnl_fwd = qnl_k
            else:
                qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(np.mod(tk,Tb)),q)
                qnl_fwd = (3./2)*qnl_k - (1./2)*qnl_km1 
                qnl_km1 = qnl_k 
            
            q = self.lu_AT.solve((1/self.dt)*q + (1/2)*cgl.A.T.dot(q) + qnl_fwd)

                
        return q
    
    
    def time_step_linear_bt(self,cgl,Qbflow,q,nsave,which):
        
        tsave = self.time[::nsave]
        Q = np.zeros((2*cgl.nx,len(tsave)))
        Q[:,0] = q
        
        k_save = 1
        for k in range (1,len(self.time)):
            
            qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(Qbflow,q)
            
            if k == 1: 
                qnl_km1 = qnl_k
                qnl_fwd = qnl_k
            else:
                qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(Qbflow,q)
                qnl_fwd = (3./2)*qnl_k - (1./2)*qnl_km1 
                qnl_km1 = qnl_k 
            
            if which == 'fwd':
                vec_rhs = (1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd
                q = self.lu_A.solve(vec_rhs)
            else:
                vec_rhs = (1/self.dt)*q + (1/2)*cgl.A.T.dot(q) + qnl_fwd
                q = self.lu_AT.solve(vec_rhs)
            
            if np.mod(k,nsave) == 0: 
                Q[:,k_save] = q
                k_save += 1
            
                
        return Q
        
        
        
                
                
                         
                
                
                
            
        
        
        
        
        
            
            
            
    