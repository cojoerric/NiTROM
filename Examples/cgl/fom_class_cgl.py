import torch
from NiTROM.PyTorch_Functions.linear_interpolation import Interp1D


class CGL: 
    
    def __init__(self,x,nu,gamma,mu0,mu2,a): 
        
        # Load grid
        self.x = x
        self.device = x.device
        self.dtype = x.dtype
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
        self.muI = torch.sparse.spdiags(self.mu,offsets=0,layout='csr')
        self.assemble_linear_operator()
        
        # Define output & input operator
        self.xb = torch.sqrt(-2*(self.mu0 - self.cu**2)/self.mu2)
        C = torch.exp(-((self.x - self.xb)/1.6)**2)
        Cr = torch.cat((C,torch.zeros(self.nx, device=self.device, dtype=self.dtype))).reshape((1,-1))
        Ci = torch.cat((torch.zeros(self.nx, device=self.device, dtype=self.dtype),C)).reshape((1,-1))
        self.C = torch.cat((Cr,Ci),axis=0)
        
        B = torch.exp(-((self.x + self.xb)/1.6)**2)
        Br = torch.cat((B,torch.zeros(self.nx, device=self.device, dtype=self.dtype))).reshape((-1,1))
        Bi = torch.cat((torch.zeros(self.nx, device=self.device, dtype=self.dtype),B)).reshape((-1,1))
        self.B = torch.cat((Br,Bi),axis=1)
        
        
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
        
        self.D = torch.sparse_csr_tensor(cols, rows, data, size=(self.nx,self.nx), device=self.device, dtype=self.dtype)
        
    
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
        
        self.DD = torch.sparse_csr_tensor(cols, rows, data, size=(self.nx,self.nx), device=self.device, dtype=self.dtype)


    def block2x2_sparse(A11, A12, A21, A22):
        # ensure COO
        A11 = A11.coalesce()
        A12 = A12.coalesce()
        A21 = A21.coalesce()
        A22 = A22.coalesce()

        r1, c1 = A11.shape
        r2, c2 = A22.shape

        def shift(A, r0, c0):
            idx = A.indices()
            val = A.values()
            idx = idx + torch.tensor([[r0], [c0]], device=idx.device)
            return idx, val

        idxs, vals = [], []
        i, v = shift(A11, 0, 0);   idxs.append(i); vals.append(v)
        i, v = shift(A12, 0, c1);  idxs.append(i); vals.append(v)
        i, v = shift(A21, r1, 0);  idxs.append(i); vals.append(v)
        i, v = shift(A22, r1, c1); idxs.append(i); vals.append(v)

        idx = torch.cat(idxs, dim=1)
        val = torch.cat(vals, dim=0)
        return torch.sparse_coo_tensor(idx, val, size=(r1+r2, c1+c2)).coalesce()
        

    def assemble_linear_operator(self):
        
        A11 = -self.nu.real*self.D + self.gamma.real*self.DD + self.muI 
        A12 = self.nu.imag*self.D - self.gamma.imag*self.DD 
        A21 = - A12 
        A22 = A11 
        
        self.A = CGL.block2x2_sparse(A11, A12, A21, A22).to_sparse_csr()
        

    def evaluate_cgl_nonlinearity(self,q1,q2,q3):
        
        nx = self.nx
        
        q1q2_real = q1[:nx]*q2[:nx] + q1[nx:]*q2[nx:] 
        
        q1q2q3_real = q1q2_real*q3[:nx] 
        q1q2q3_imag = q1q2_real*q3[nx:] 
        
        return -self.a*torch.cat((q1q2q3_real,q1q2q3_imag),axis=0)
    
    
    def evaluate_right_hand_side(self,q):
        return self.A.dot(q) + self.evaluate_cgl_nonlinearity(q,q,q)
    
    
    def evaluate_cgl_linearized_nonlinearity(self,Q,q):
        
        nx = self.nx 
        
        qreal = (3*Q[:nx]**2 + Q[nx:]**2)*q[:nx] + 2*Q[:nx]*Q[nx:]*q[nx:]
        qimag = 2*Q[:nx]*Q[nx:]*q[:nx] + (3*Q[nx:]**2 + Q[:nx]**2)*q[nx:]
        
        return -self.a*torch.cat((qreal,qimag),axis=0)
    

    def evaluate_cgl_adjoint_rhs(self,Q,q):
        return self.A.T.dot(q) + self.evaluate_cgl_linearized_nonlinearity(Q,q)
    
    
    def assemble_random_periodic_vector(self,freqs,time):
        
        nf = len(freqs)
        modes = torch.rand(2*self.nx,nf) + 1j*torch.rand(2*self.nx,nf)
        modes[:,0] = modes[:,0].real
        
        # Normalize the modes so that the forcing is unit norm
        val = torch.dot(modes[:,0],modes[:,0])
        for k in range (1,nf):  val += 2*torch.dot(modes[:,k].conj(),modes[:,k])
        modes = torch.sqrt(1/val)*modes
        
        f = torch.outer(modes[:,0],torch.ones(len(time))).real
        for k in range (1,len(freqs)):  f += 2*torch.outer(modes[:,k],torch.exp(1j*freqs[k]*time)).real
            
        return f.real
    
    
    """
        Functions that conform with those expected by the TrOOP during the 
        optimization
    """
    
    def evaluate_fom_dynamics(self,t,q,u):
        
        f = u.copy() if hasattr(u,"__len__") == True else u(t)
        if torch.linalg.norm(q) >= 1e4:    vec = 0*q
        else:                           vec = self.evaluate_right_hand_side(q) + f
        
        return vec
    
    def evaluate_fom_adjoint(self,t,q,fQ):
        
        if torch.linalg.norm(q) >= 1e4:    vec = 0*q
        else:                           vec = self.evaluate_cgl_adjoint_rhs(fQ(t),q)
        
        return vec
    
    def compute_output(self,q):
        return self.C@q
    
    def compute_output_derivative(self,q):
        return self.C
    
    def assemble_petrov_galerkin_tensors(self,Phi,Psi):
        
        n, r = Phi.shape
        PhiF = Phi@torch.linalg.inv(Psi.T@Phi)
        
        Br = Psi.T@self.B
        Cr = self.C@PhiF
        Ar = Psi.T@(self.A.dot(PhiF))
        Hr = torch.zeros((r,r,r,r), device=self.device, dtype=self.dtype)
        
        for i in range (r):
            for j in range (r):
                for k in range (r): 
                    Hr[:,i,j,k] = Psi.T@(self.evaluate_cgl_nonlinearity(PhiF[:,i],PhiF[:,j],PhiF[:,k]))
        
        return (Ar, Hr), (Br, Cr)
    

    
class time_step_cgl: 
    
    def __init__(self,cgl,time): 
        
        self.time = time
        self.dt = time[1] - time[0]
        self.Id = torch.sparse_csr_tensor(
            torch.arange(2*cgl.nx, device=cgl.device, dtype=torch.int64).unsqueeze(0).expand(2, -1),
            torch.ones(2*cgl.nx, device=cgl.device, dtype=cgl.dtype),
            size=(2*cgl.nx, 2*cgl.nx)
        )
        self.A = (1/self.dt)*self.Id - (1/2)*cgl.A
        self.AT = (1/self.dt)*self.Id - (1/2)*cgl.A.T
            
            
    
    def time_step(self,cgl,q,nsave,*argv): 
        
        tsave = self.time[::nsave]
        Q = torch.zeros((2*cgl.nx,len(tsave)), device=cgl.device, dtype=cgl.dtype)
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
                
                q = torch.sparse.spsolve(self.A, (1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd)
                
                if k % nsave == 0: 
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
                
                q = torch.sparse.spsolve(self.A, (1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd + fU(self.time[k-1] % TfU))
                
                if k % nsave == 0: 
                    Q[:,k_save] = q
                    k_save += 1
            
                    
        return Q, cgl.C@Q, tsave
    
    
    def time_step_linearized(self,cgl,Qbflow,tbflow,q,*argv):
        
        fQ = Interp1D(tbflow, Qbflow, extrapolate=True)
        Tb = tbflow[-1] + (tbflow[1] - tbflow[0])
        
        forcing_flag = 0
        if len(argv) > 0: 
            fF = argv[0]
            Tf = argv[1]
            forcing_flag = 1
        
        for k in range (1,len(self.time)):
            
            tk = self.time[k] - 1
            qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(tk % Tb),q)
            
            if k == 1: 
                qnl_km1 = qnl_k
                qnl_fwd = qnl_k
            else:
                qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(tk % Tb),q)
                qnl_fwd = (3./2)*qnl_k - (1./2)*qnl_km1 
                qnl_km1 = qnl_k 
                
            vec_rhs = (1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd
            if forcing_flag == 1:
                vec_rhs = (1/self.dt)*q + (1/2)*cgl.A.dot(q) + qnl_fwd + fF(tk % Tf)
                
            q = torch.sparse.spsolve(self.A, vec_rhs)
            
                
        return q
    
    
    def time_step_adjoint(self,cgl,Qbflow,tbflow,q):
        
        fQ = Interp1D(tbflow, torch.fliplr(Qbflow), extrapolate=True)
        Tb = tbflow[-1] + (tbflow[1] - tbflow[0])

        for k in range (1,len(self.time)):
            
            tk = self.time[k] - 1
            qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(tk % Tb),q)
            
            if k == 1: 
                qnl_km1 = qnl_k
                qnl_fwd = qnl_k
            else:
                qnl_k = cgl.evaluate_cgl_linearized_nonlinearity(fQ(tk % Tb),q)
                qnl_fwd = (3./2)*qnl_k - (1./2)*qnl_km1 
                qnl_km1 = qnl_k 
            
            q = torch.sparse.spsolve(self.A.T, (1/self.dt)*q + (1/2)*cgl.A.T.dot(q) + qnl_fwd)

                
        return q
    
    
    def time_step_linear_bt(self,cgl,Qbflow,q,nsave,which):
        
        tsave = self.time[::nsave]
        Q = torch.zeros((2*cgl.nx,len(tsave)), device=cgl.device, dtype=cgl.dtype)
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
                q = torch.sparse.spsolve(self.A, vec_rhs)
            else:
                vec_rhs = (1/self.dt)*q + (1/2)*cgl.A.T.dot(q) + qnl_fwd
                q = torch.sparse.spsolve(self.A.T, vec_rhs)
            
            if k % nsave == 0: 
                Q[:,k_save] = q
                k_save += 1
            
                
        return Q
        
        
def balanced_truncation(cgl,cgl_tstep,Qbflow,nsave,r):
    
    nsnaps = len(cgl_tstep.time[::nsave])
    Qfwd = torch.zeros((cgl.B.shape[0],cgl.B.shape[-1]*nsnaps), device=cgl.device, dtype=cgl.dtype)
    for k in range (cgl.B.shape[-1]):
        print("Running forward %d/%d"%(k+1,cgl.B.shape[-1]))
        idx0 = k*nsnaps
        idx1 = (k+1)*nsnaps 
        Qfwd[:,idx0:idx1] = cgl_tstep.time_step_linear_bt(cgl,Qbflow,cgl.B[:,k],nsave,'fwd')
        
    
    Qadj = torch.zeros((cgl.B.shape[0],cgl.C.shape[0]*nsnaps), device=cgl.device, dtype=cgl.dtype)
    for k in range (cgl.C.shape[0]):
        print("Running adjoint %d/%d"%(k+1,cgl.C.shape[0]))
        idx0 = k*nsnaps
        idx1 = (k+1)*nsnaps 
        Qadj[:,idx0:idx1] = cgl_tstep.time_step_linear_bt(cgl,Qbflow,cgl.C[k,],nsave,'adj') 
        
    
    u, s, v = torch.linalg.svd(Qadj.T@Qfwd,full_matrices=False) 
    v = v.T
    Phi = Qfwd@v[:,:r]@torch.diag(1./torch.sqrt(s[:r])) 
    Psi = Qadj@u[:,:r]@torch.diag(1./torch.sqrt(s[:r]))  
    
    return Phi, Psi