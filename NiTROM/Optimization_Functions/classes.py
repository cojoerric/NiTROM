import numpy as np 
from mpi4py import MPI
from itertools import combinations
from string import ascii_lowercase as ascii

class mpi_pool:

    def __init__(self,comm,n_traj,fname_traj,fname_time,**kwargs):
        """
        Initialize MPI pool and load distributed training data.

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator.
        n_traj : int
            Total number of trajectories to load from disk.
        fname_traj : str
            Filename pattern used to load each trajectory, e.g. 'traj_%03d.npy'.
        fname_time : str
            Filename for the time vector, e.g. 'time.txt'.
        **kwargs : optional
            fname_weights : str, optional
                Filename pattern for weights, e.g. 'weight_%03d.npy'.
            fname_steady_forcing : str, optional
                Filename pattern for steady forcing, e.g. 'forcing_%03d.npy'.
            fname_derivs : str, optional
                Filename pattern for time derivatives, e.g. 'fname_derivs_%03d.npy'.

        Notes
        -----
        Each MPI process instantiates its own mpi_pool and loads only the subset
        of trajectories assigned to that rank. The full dataset is distributed
        across the MPI pool.
        """

        self.comm = comm                            # MPI communicator
        self.size = self.comm.Get_size()            # Total number of processes
        self.rank = self.comm.Get_rank()            # Id of the current process

        self.n_traj = n_traj                        # Total number of training trajectories
        if self.size > self.n_traj:
            raise ValueError ("You have more MPI processes than trajectories!")
        else:
            if self.rank == 0:
                print("Hello, you are running NiTROM with %d MPI processors."%self.size)
        
        self.my_n_traj = self.n_traj//self.size     # Number of trajectories owned by process self.rank
        self.my_n_traj += 1 if np.mod(self.n_traj,self.size) > self.rank else 0


        # Vectors used for future MPI communications
        self.counts = np.zeros(self.size,dtype=np.int64)    
        self.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[self.counts,MPI.INT])
        self.disps = np.concatenate(([0],np.cumsum(self.counts)[:-1])) 
        
        
        # Load data from file
        self.load_trajectories(fname_traj)
        self.time = np.load(fname_time)
        self.load_weights(kwargs)
        self.load_steady_forcing(kwargs)
        self.load_time_derivatives(kwargs)
        
        
    def load_trajectories(self,fname_traj):
        """
        Load trajectory files for the local MPI rank.

        Parameters
        ----------
        fname_traj : str
            Filename pattern for trajectories assigned to this process.

        Notes
        -----
        The method populates:
          - self.fnames_traj : list of filenames loaded by this rank
          - self.X : numpy array of shape (my_n_traj, N, n_snapshots)
          - self.N, self.n_snapshots : dimensions inferred from the first file
        """
        self.fnames_traj = [fname_traj%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
        X = [np.load(self.fnames_traj[k]) for k in range (self.my_n_traj)]
        self.N, self.n_snapshots = X[0].shape
        self.X = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
        for k in range (self.my_n_traj): self.X[k,] = X[k]
        
    def load_weights(self,kwargs):
        """
        Load per-trajectory weights if provided.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed into __init__; looks for 'fname_weights'.

        Notes
        -----
        If no weights file is provided, weights default to ones.
        """        
        fname_weights = kwargs.get('fname_weights',None)
        self.weights = np.ones(self.my_n_traj)
        if fname_weights != None:
            self.fnames_weights = [fname_weights%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            weights = [np.load(self.fnames_weights[k]) for k in range (self.my_n_traj)]
            self.weights = np.zeros(self.my_n_traj)
            for k in range (self.my_n_traj): self.weights[k] = weights[k]
            
    def load_steady_forcing(self,kwargs):
        """
        Load steady forcing vectors for each local trajectory if provided.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed into __init__; looks for 'fname_steady_forcing'.

        Notes
        -----
        Populates self.F with shape (N, my_n_traj). If no files are provided,
        self.F remains a zero array.
        """        
        fname_forcing = kwargs.get('fname_steady_forcing',None)
        self.F = np.zeros((self.N,self.my_n_traj))
        if fname_forcing != None:
            self.fnames_forcing = [(fname_forcing)%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            for k in range (self.my_n_traj):  self.F[:,k] = np.load(self.fnames_forcing[k])
    
    def load_time_derivatives(self,kwargs):
        """
        Load precomputed time derivatives for trajectories if provided.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed into __init__; looks for 'fname_derivs'.

        Notes
        -----
        Populates self.dX with shape (my_n_traj, N, n_snapshots) when files exist.
        """        
        fname_deriv = kwargs.get('fname_derivs',None)
        if fname_deriv != None:
            self.fnames_deriv = [fname_deriv%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            dX = [np.load(self.fnames_deriv[k]) for k in range (self.my_n_traj)]
            self.dX = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
            for k in range (self.my_n_traj): self.dX[k,] = dX[k]
        
class optimization_objects:

    def __init__(self,mpi_pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp,**kwargs):
        """
        Prepare training data information and optimization objects to be passed to the optimizer.

        Parameters
        ----------
        mpi_pool : mpi_pool
            Instance of the mpi_pool class containing distributed data.
        which_trajs : array_like
            Indices selecting trajectories from mpi_pool.X to include in this batch. 
            Useful for stochastic gradient descent.
        which_times : array_like
            Indices selecting time snapshots to include from each trajectory. 
            Useful if we want to start training on short trajectories and then progressively 
            extend the length of the trajectories.
        leggauss_deg : int
            Number of Gauss–Legendre quadrature points for integral approximations. 
            For further details, see Prop. 2.1 in NiTROM arXiv paper
        nsave_rom : int
            Number of ROM snapshots stored between successive FOM snapshots.
        poly_comp : sequence of int
            Polynomial components of the ROM; e.g. [1, 2] for linear and quadratic terms.

            .. math::
            f_r = A_r\hat{z} + B_ru + H_r:\hat{z}\hat{z}^T + L_r:\hat{z}u^T + \ldots

            
        **kwargs : optional
            which_fix : {'fix_bases', 'fix_tensors', 'fix_none'}, default 'fix_none'
                Which quantities to keep fixed during optimization.
            stab_promoting_pen : float, optional
                L2 regularization coefficient for stability-promoting penalty.
            stab_promoting_tf : float, optional
                Final time used by the stability-promoting penalty.
            stab_promoting_ic : array_like, optional
                Initial condition (random) normalized vector used to probe stability penalty.

        Raises
        ------
        ValueError
            If invalid which_fix provided, or if required stability penalty arguments are missing.

        Notes
        -----
        This class slices mpi_pool data according to which_trajs and which_times,
        rescales trajectory weights so the cost measures average error over snapshots
        and trajectories, and generates einsum subscripts for efficient tensor contractions.
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
        self.weights *= np.sum(counts)*self.n_snapshots
        
        # Parse the keyword arguments
        self.which_fix = kwargs.get('which_fix','fix_none')
        if self.which_fix not in ['fix_tensors','fix_bases','fix_none']:
            raise ValueError ("which_fix must be fix_none, fix_tensors or fix_bases")
            
        self.l2_pen = kwargs.get('stab_promoting_pen',None)
        self.pen_tf = kwargs.get('stab_promoting_tf',None)
        self.randic = kwargs.get('stab_promoting_ic',None)
        
        if self.l2_pen != None and self.pen_tf == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a value for stab_promoting_tf")
                              
        if self.l2_pen != None and self.randic == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a random ic vector of the same \
                              size as the ROM")
                              
        if self.l2_pen != None and 1 not in self.poly_comp:
            raise ValueError ("The penalty is currently implemented for the linear term \
                              in the rom dynamics. You have no linear term.")
                              
        if self.randic != None: 
            self.randic /= np.linalg.norm(self.randic)
            self.randic = self.randic.reshape(-1)
            
        
    
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
            u = u.copy() if hasattr(u,"__len__") == True else u(t)
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
                
                combs = list(combinations(self.einsum_ss[i][1:],r=k-1))
                operands = [operators[i]] + [fq(t) for _ in range(k-1)]
                for comb in combs:
                    equation = [self.einsum_ss[i][0]] + list(comb)
                    equation = ",".join(equation)
                    
                    J += np.einsum(equation,*operands)
                    
            dzdt = J.T@z 
            
        return dzdt
    
    
        
    

        
        
        
        
        
        
        
        
        
