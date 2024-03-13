import numpy as np 
from mpi4py import MPI
from itertools import combinations
from string import ascii_lowercase as ascii

class mpi_pool:

    def __init__(self,comm,n_traj,fname_traj,fname_time,**kwargs):
    #(self,comm,n_traj,fname_root,format,file_type,fname_time,fname_root_weights,*argv):
        
        
        """ 
        This class contains all the info regarding the MPI pool that will be used 
        during optimization. It also loads the training data from disk. Every process
        (with Id "self.rank") owns its own instance of this class and its own chunk of 
        the training data. I.e., the whole training data set is distributed
        across the whole MPI pool.  
        
        comm:           MPI Communicator
        n_traj:         total number of trajectories we wish to load from disk
        fname_traj:     e.g., 'traj_%03d.txt' (string used to load each trajectory)
        fname_time:     e.g., 'time.txt' (time vector at which we save snapshots)
        
        Optional keyword arguments:
            fname_weights:          e.g., 'weight_%03d.txt' 
            fname_steady_forcing:   e.g., 'forcing_%03d.txt'
            
        """

        self.comm = comm                            # MPI communicator
        self.size = self.comm.Get_size()            # Total number of processes
        self.rank = self.comm.Get_rank()            # Id of the current process

        self.n_traj = n_traj                        # Total number of training trajectories
        if self.size > self.n_traj:
            raise ValueError ("You have more MPI processes than trajectories!")
        
        self.my_n_traj = self.n_traj//self.size     # Number of trajectories owned by process self.rank
        self.my_n_traj += 1 if np.mod(self.n_traj,self.size) > self.rank else 0


        # Vectors used for future MPI communications
        self.counts = np.zeros(self.size,dtype=np.int64)    
        self.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[self.counts,MPI.INT])
        self.disps = np.concatenate(([0],np.cumsum(self.counts)[:-1])) 
        
        
        # Load data from file
        self.load_trajectories(fname_traj)
        self.load_weights(kwargs)
        self.load_steady_forcing(kwargs)
        self.time = np.loadtxt(fname_time)
        
        
    def load_trajectories(self,fname_traj):
        
        self.fnames_traj = [fname_traj%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
        X = [np.loadtxt(self.fnames_traj[k]) for k in range (self.my_n_traj)]
        self.N, self.n_snapshots = X[0].shape
        self.X = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
        for k in range (self.my_n_traj): self.X[k,] = X[k]
        
    def load_weights(self,kwargs):
        
        fname_weights = kwargs.get('fname_weights',None)
        self.weights = np.ones(self.my_n_traj)
        if fname_weights != None:
            self.fnames_weights = [fname_weights%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            weights = [np.loadtxt(self.fnames_weights[k]) for k in range (self.my_n_traj)]
            self.weights = np.zeros(self.my_n_traj)
            for k in range (self.my_n_traj): self.weights[k] = weights[k]
            
    def load_steady_forcing(self,kwargs):
        
        fname_forcing = kwargs.get('fname_steady_forcing',None)
        self.F = np.zeros((self.N,self.my_n_traj))
        if fname_forcing != None:
            self.fnames_forcing = [(fname_forcing)%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            for k in range (self.my_n_traj):  self.F[:,k] = np.loadtxt(self.fnames_forcing[k])
        
        


class optimization_objects:

    def __init__(self,mpi_pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp,**kwargs):

        """ 
        This class contains the training data information that will get passed to pymanopt. 
        
        mpi_pool:       an instance of the mpi_pool class
        which_trajs:    array of integers to extract a subset of the trajectories contained in 
                        mpi_pool.X. Useful if we end up using stochastic gradient descent
        which_times:    arrays of integers to extract a subset of a trajectory owned by mpi_pool. Useful
                        if we want to start training on short trajectories and then progressively extend 
                        the length of the trajectories
        leggauss_deg:   number of Gauss-Legendre quadrature points used to approximate the integrals
                        in the gradient (see Prop. 2.1 in NiTROM arXiv paper)
        nsave_rom:      number of ROM snapshots to store in between two adjacent FOM snapshots
        poly_comp:      component of the polynomial ROM (e.g., [1,2] is a quadratic model with both first and
                                                         second order components) 

        Optional keyword arguments:
            which_fix:      one of fix_bases, fix_tensors or fix_none (default is fix_none)
        """
        
        
        self.X = mpi_pool.X[which_trajs,:,:]      
        self.X = self.X[:,:,which_times]
        self.F = mpi_pool.F[:,which_trajs]
        self.time = mpi_pool.time[which_times]
        self.weights = mpi_pool.weights[which_trajs]

        self.my_n_traj, _, self.n_snapshots = self.X.shape
        self.leggauss_deg = leggauss_deg
        self.nsave_rom = nsave_rom
        self.poly_comp = poly_comp
        self.generate_einsum_subscripts()
        
        # Count the total number of trajectories in this batch and
        # scale the weight accordingly so that the cost function measures
        # the average error over snapshots and trajectories. (Notice that 
        # if all trajectories are loaded, then np.sum(counts) = mpi_pool.n_traj)
        counts = np.zeros(mpi_pool.size,dtype=np.int64)    
        mpi_pool.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[counts,MPI.INT])
        self.weights *= np.sum(counts)
        
        self.which_fix = kwargs.get('which_fix','fix_none')
        if self.which_fix not in ['fix_tensors','fix_bases','fix_none']:
            raise ValueError ("which_fix must be fix_none, fix_tensors or fix_bases")
    
    
    def generate_einsum_subscripts(self):
        """
            Generates the indices for the einsum evaluation of the 
            right-hand side and the adjoint
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii[:k+1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        
        self.einsum_ss = tuple(ss)
        
        

    def evaluate_rom_rhs(self,t,z,u,*operators,**kwargs):
        """
            Function that can be fed into scipys solve_ivp. 
            t:          time instance
            z:          state vector
            u:          a steady forcing vector
            operators:  (A2,A3,A4,...)
            
            Optional keyword arguments:
                'forcing_interp':   a scipy interpolator f that gives us a forcing f(t)
        """
        
        if np.linalg.norm(z) >= 1e4:    
            dzdt = 0.0*z 
        else:
            f = kwargs.get('forcing_interp',None)
            f = f(t) if f != None else np.zeros(len(z))
            dzdt = u + f
            for (i, k) in enumerate(self.poly_comp):
                equation = ",".join(self.einsum_ss[i])
                operands = [operators[i]] + [z for _ in range(k)]
                dzdt += np.einsum(equation,*operands)
        
        return dzdt
    
    
    def evaluate_rom_adjoint(self,t,z,fq,*operators):
        """
            Function that can be fed into scipys solve_ivp. 
            t:          time instance
            z:          state vector
            fq:         interpolator (from scipy.interpolate) to evaluate the
                        base flow at time t
            operators:  (A2,A3,A4,...)
        """
        
        if np.linalg.norm(z) >= 1e4:
            dzdt = 0.0*z
        else:
            J = np.zeros((len(z),len(z)))
            for (i, k) in enumerate(self.poly_comp):
                
                combs = list(combinations(self.einsum_ss[i][1:], r=k-1))
                operands = [operators[i]] + [fq(t) for _ in range(k-1)]
                for comb in combs:
                    equation = [self.einsum_ss[i][0]] + list(comb)
                    equation = ",".join(equation)
                    
                    J += np.einsum(equation,*operands)
                    
            dzdt = J.T@z 
            
        return dzdt
    

        
        
        
        
        
        
        
        
        
