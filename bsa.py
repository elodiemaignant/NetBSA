import numpy as np
from itertools import combinations
from scipy.linalg import expm, logm
from scipy.optimize import LinearConstraint, minimize

class NetBSA:
    """Class for sample-limited barycentric subspace analysis (BSA) of 
    network-valued data.
    
    It implements several variants of the barycentric projection and 
    sample-limited BSA.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    k_ref : int
        Number of reference networks
        
    Attributes
    ----------
    ref_samples : array-like, shape=[k_ref]
        Indices of the reference networks.
    ref_networks : array-like, shape=[k_ref, n_nodes, n_nodes]
        Reference networks.    
    
    References
    ----------
    .. [P2016] Pennec, X. (2018). "Barycentric subspace analysis on manifolds". 
    Ann. Statist. 46(6A): 2711-2746. 
    https://doi.org/10.1214/17-AOS1636
    """
    
    def __init__(self, n_nodes, k_ref):
        self.n_nodes=n_nodes
        self.k_ref=k_ref

    def _barycentric_projection(self, network, ref_networks, tol=1E-12, cons_tol=1E-12, max_it=1000): #used to be max_it=100
        """Compute the projection of a network onto the barycentric subspace of 
        some reference networks.
        
        Parameters
        ----------
        network : array-like, shape=[n_nodes, n_nodes]
            Network to project.
        ref_networks : array-like, shape=[k_ref, n_nodes, n_nodes]
            Reference networks of the barycentric subspace to project onto.
        tol : float
            Tolerance for termination.
        cons_tol : float
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        weights : array-like, shape=[k_ref]
            Barycentric weights of the projection.
        error : float
            Squared projection error.
        """
        k_ref = ref_networks.shape[0]
        
        spec = np.linalg.eigvalsh(network)
        ref_spec = np.linalg.eigvalsh(ref_networks)

        #constraints
        squared_error = lambda w: np.dot(w @ ref_spec - spec, w @ ref_spec - spec)
        jac = lambda w: 2. * (w @ ref_spec - spec) @ np.transpose(ref_spec)

        sum_to_one = LinearConstraint(A=np.ones(k_ref), lb=1.-cons_tol, ub=1.+cons_tol)
    
        A = np.zeros((self.n_nodes - 1, k_ref))
        for i in range(self.n_nodes - 1): A[i, :] = ref_spec[:, i+1] - ref_spec[:, i]
        in_the_cone = LinearConstraint(A=A, lb=-cons_tol, ub=np.inf)

        #initialization
        w0 = np.ones(k_ref) / k_ref  
        
        res = minimize(fun=squared_error, x0=w0, method='SLSQP', jac=jac, 
                       constraints=[sum_to_one, in_the_cone], tol=tol, 
                       options={'maxiter' : max_it})
        
        if not res.success: print("The projection did not converge.", res.x, A @ res.x)
        
        return(res.x, res.fun)

    def _convex_barycentric_projection(self, network, ref_networks, tol=1E-12, cons_tol=1E-12, max_it=1000):
        """Compute the projection of a network onto the convex barycentric 
        subspace of some reference networks.
        
        Parameters
        ----------
        network : array-like, shape=[n_nodes, n_nodes]
            Network to project.
        ref_networks : array-like, shape=[k_ref, n_nodes, n_nodes]
            Reference networks.
        tol : float
            Tolerance for termination.
        cons_tol : float
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        weights : array-like, shape=[k_ref]
            Barycentric weights of the projection.
        error : float
            Squared projection error.
        """
        k_ref = ref_networks.shape[0]
        
        spec = np.linalg.eigvalsh(network)
        ref_spec = np.linalg.eigvalsh(ref_networks)
        
        squared_error = lambda w: np.dot(w @ ref_spec - spec, w @ ref_spec - spec)
        jac = lambda w: 2. * (w @ ref_spec - spec) @ np.transpose(ref_spec)

        sum_to_one = LinearConstraint(A=np.ones(k_ref), lb=1.-cons_tol, ub=1.+cons_tol)
    
        are_positive = LinearConstraint(A=np.eye(k_ref), lb=-cons_tol, ub=np.inf)
    
        w0 = np.zeros(len(ref_networks))
        w0[np.argmin(np.linalg.norm(ref_networks - network, axis=(1, 2)))] = 1.
        
        res = minimize(fun=squared_error, x0=w0, method='SLSQP', jac=jac, 
                       constraints=[sum_to_one, are_positive], tol=tol, 
                       options={'maxiter' : max_it})
       
        # if not res.success: print((res.x > 1E-3) * res.jac)
    
        return(res.x, res.fun)

    def _bsa(self, networks, k_ref):
        """Perform sample-limited barycentric subspace analysis of 
        a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
            Dataset.
        k_ref : int
            Number of reference networks
        tol : float
            Tolerance for termination.
        cons_tol : float
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        ref_samples : list of int of length k_ref
            Indices of the reference networks of the optimal barycentric 
            subspace.
        weights : array-like, shape=[n_samples, k_ref]
            Barycentric weights of the projection of the dataset onto the 
            optimal subspace.
        total_error : float
            Total squared projection error.
        """
        n_samples = networks.shape[0]
        
        ref_samples = None
        weights = None
        total_error = np.inf
        
        for s in combinations(range(n_samples), r=k_ref):
            ref_samples_s = list(s)
            weights_s = []
            total_error_s = 0
            i = 0
            
            while total_error_s < total_error and i < n_samples:
                weights_i, error_i = self._barycentric_projection(networks[i], networks[ref_samples_s])
                total_error_s += error_i
                weights_s.append(weights_i)
                i += 1
            
            if total_error_s < total_error:
                ref_samples = ref_samples_s
                weights = weights_s
                total_error = total_error_s

        return(ref_samples, np.array(weights), np.array(total_error))
    
    def _forward_bsa(self, networks, k_max=None):
        """Perform sample-limited forward barycentric subspace analysis of 
        a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
            Dataset.
        k_max : int
            Number of reference networks of the largest barycentric subspace of 
            the nested sequence. 
        tol : float
            Tolerance for termination.
        cons_tol : float
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        ref_samples_seq : list of lists of length k_max
            Indices of the reference networks of the optimal nested barycentric 
            subspaces.
        weights_seq : list of arrays of length k_max
            Barycentric weights of the projection onto each subspace of the 
            optimal nested sequence.
        total_error_seq : list of floats of length k_max
            Total squared projection errors.
        """
        n_samples = networks.shape[0]
        if k_max is None: k_max = n_samples
        
        #median
        median = None
        total_error = np.inf
        
        for i in range(n_samples):
            total_error_i = np.linalg.norm(np.linalg.eigvalsh(networks) 
                                           - np.linalg.eigvalsh(networks[i])) ** 2
            
            if total_error_i < total_error: 
                median = i
                total_error = total_error_i
        
        #forward
        ref_samples_seq = [[median]]
        weights_seq = [np.array(n_samples * [i==median for i in range(n_samples)])]
        total_error_seq = [total_error]
        
        ref_samples = [median]
        not_a_ref = [i for i in range(n_samples) if i != median]
        
        while len(ref_samples) < k_max:
            print(len(ref_samples))
            weights = None
            total_error = np.inf
            sample_to_add = None
            
            for i in not_a_ref:
                ref_samples_i = ref_samples + [i]
                weights_i = []
                total_error_i = 0
                j = 0
    
                while total_error_i < total_error and j < n_samples:
                    weights_j, error_j = self._barycentric_projection(networks[j], networks[ref_samples_i])
                    weights_i.append(weights_j)
                    total_error_i += error_j
                    j += 1
    
                if total_error_i < total_error:
                    sample_to_add = i
                    weights = weights_i
                    total_error = total_error_i
                    
            ref_samples.append(sample_to_add)
            not_a_ref.remove(sample_to_add)
            
            ref_samples_seq.append(list(np.copy(ref_samples)))
            weights_seq.append(np.copy(weights))
            total_error_seq.append(np.copy(total_error))
                
        return(ref_samples_seq, weights_seq, total_error_seq)
    
    def _backward_bsa(self, networks, ref_samples_init=None):
        """Perform sample-limited backward barycentric subspace analysis of 
        a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
            Dataset.
        ref_samples_init : list of int of len k_init
            Indices of the reference networks of the initial (largest) 
            barycentric subspace.
        tol : float.
            Tolerance for termination.
        cons_tol : float.
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        ref_samples_seq : list of lists of length k_init
            Indices of the reference networks of the optimal nested barycentric 
            subspaces.
        weights_seq : list of arrays of length k_init
            Barycentric weights of the projection onto each subspace of the 
            optimal nested sequence.
        total_error_seq : list of floats of length k_init
            Total squared projection errors.
        """
        n_samples = networks.shape[0]
        if ref_samples_init is None: ref_samples_init = list(range(n_samples))
        
        #initialization
        weights_init = []
        total_error_init = 0.
        
        for i in range(n_samples):
            weights_i, error_i = self._barycentric_projection(networks[i], networks[ref_samples_init])
            weights_init.append(weights_i)
            total_error_init += error_i
        
        #backward
        ref_samples_seq = [ref_samples_init]
        weights_seq = [weights_init]
        total_error_seq = [total_error_init]

        ref_samples = list(np.copy(ref_samples_init))
        
        while len(ref_samples) > 1:
            print(len(ref_samples))
            weights = None
            total_error = np.inf
            sample_to_remove = None
            
            for i in ref_samples:
                ref_samples_i = [j for j in ref_samples if j != i]
                weights_i = []
                total_error_i = 0
                j = 0
    
                while total_error_i < total_error and j < n_samples:
                    weights_j, error_j = self._barycentric_projection(networks[j], networks[ref_samples_i])
                    weights_i.append(weights_j)
                    total_error_i += error_j
                    j += 1
    
                if total_error_i < total_error:
                    sample_to_remove = i
                    weights = weights_i
                    total_error = total_error_i
            
            ref_samples.remove(sample_to_remove)
            
            ref_samples_seq.append(list(np.copy(ref_samples)))
            weights_seq.append(np.copy(weights))
            total_error_seq.append(np.copy(total_error))
            
        ref_samples_seq.reverse()
        weights_seq.reverse()
        total_error_seq.reverse()
                
        return(ref_samples_seq, weights_seq, total_error_seq)

    def _convex_bsa(self, networks, k_ref):
        """Perform sample-limited convex barycentric subspace analysis of 
        a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
            Dataset.
        k_ref : int
            Number of reference networks
        tol : float
            Tolerance for termination.
        cons_tol : float
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        ref_samples : list of int of length k_ref
            Indices of the reference networks of the optimal barycentric 
            subspace.
        weights : array-like, shape=[n_samples, k_ref]
            Barycentric weights of the projection of the dataset onto the 
            optimal subspace.
        total_error : float
            Total squared projection error.
        """
        n_samples = networks.shape[0]
        
        ref_samples = None
        weights = None
        total_error = np.inf
        
        for s in combinations(range(n_samples), r=k_ref):
            ref_samples_s = list(s)
            weights_s = []
            total_error_s = 0
            i = 0
            
            while total_error_s < total_error and i < n_samples:
                weights_i, error_i = self._convex_barycentric_projection(networks[i], networks[ref_samples_s])
                total_error_s += error_i
                weights_s.append(weights_i)
                i += 1
            
            if total_error_s < total_error:
                ref_samples = ref_samples_s
                weights = weights_s
                total_error = total_error_s

        return(ref_samples, np.array(weights), np.array(total_error))
    
    def _forward_convex_bsa(self, networks, k_max=None):
        """Perform sample-limited forward convex barycentric subspace analysis 
        of a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
            Dataset.
        k_max : int
            Number of reference networks of the largest barycentric subspace of 
            the nested sequence. 
        tol : float
            Tolerance for termination.
        cons_tol : float
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        ref_samples_seq : list of lists of length k_max
            Indices of the reference networks of the optimal nested barycentric 
            subspaces.
        weights_seq : list of arrays of length k_max
            Barycentric weights of the projection onto each subspace of the 
            optimal nested sequence.
        total_error_seq : list of floats of length k_max
            Total squared projection errors.
        """
        n_samples = networks.shape[0]
        if k_max is None: k_max = n_samples
        
        #initialization
        ref_samples_init, weights_init, total_error_init = self._convex_bsa(networks, 2)
        
        #forward
        ref_samples_seq = [ref_samples_init]
        weights_seq = [weights_init]
        total_error_seq = [total_error_init]
        
        ref_samples = list(np.copy(ref_samples_init))
        not_a_ref = [i for i in range(n_samples) if i not in ref_samples_init]
        
        while len(ref_samples) < k_max:
            print(len(ref_samples))
            weights = None
            total_error = np.inf
            sample_to_add = None
            
            for i in not_a_ref:
                ref_samples_i = ref_samples + [i]
                weights_i = []
                total_error_i = 0
                j = 0
    
                while total_error_i < total_error and j < n_samples:
                    weights_j, error_j = self._convex_barycentric_projection(networks[j], networks[ref_samples_i])
                    weights_i.append(weights_j)
                    total_error_i += error_j
                    j += 1
    
                if total_error_i < total_error:
                    sample_to_add = i
                    weights = weights_i
                    total_error = total_error_i
                    
            ref_samples.append(sample_to_add)
            not_a_ref.remove(sample_to_add)
            
            ref_samples_seq.append(list(np.copy(ref_samples)))
            weights_seq.append(np.copy(weights))
            total_error_seq.append(np.copy(total_error))
                
        return(ref_samples_seq, weights_seq, total_error_seq)
    
    def _backward_convex_bsa(self, networks, ref_samples_init=None):
        """Perform sample-limited backward convex barycentric subspace analysis 
        of a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
            Dataset.
        ref_samples_init : list of int of len k_init
            Indices of the reference networks of the initial (largest) 
            barycentric subspace.
        tol : float.
            Tolerance for termination.
        cons_tol : float.
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        
        Returns
        -------
        ref_samples_seq : list of lists of length k_init
            Indices of the reference networks of the optimal nested barycentric 
            subspaces.
        weights_seq : list of arrays of length k_init
            Barycentric weights of the projection onto each subspace of the 
            optimal nested sequence.
        total_error_seq : list of floats of length k_init
            Total squared projection errors.
        """
        n_samples = networks.shape[0]
        if ref_samples_init is None: ref_samples_init = list(range(n_samples))

        #initialization
        weights_init = []
        total_error_init = 0.
        
        for i in range(n_samples):
            weights_i, error_i = self._convex_barycentric_projection(networks[i], networks[ref_samples_init])
            weights_init.append(weights_i)
            total_error_init += error_i
        
        #backward
        ref_samples_seq = [ref_samples_init]
        weights_seq = [weights_init]
        total_error_seq = [total_error_init]

        ref_samples = list(np.copy(ref_samples_init))
        
        while len(ref_samples) > 1:
            print(len(ref_samples))
            weights = None
            total_error = np.inf
            sample_to_remove = None
            
            for i in ref_samples:
                ref_samples_i = [j for j in ref_samples if j != i]
                weights_i = []
                total_error_i = 0
                j = 0
    
                while total_error_i < total_error and j < n_samples:
                    weights_j, error_j = self._convex_barycentric_projection(networks[j], networks[ref_samples_i])
                    weights_i.append(weights_j)
                    total_error_i += error_j
                    j += 1
    
                if total_error_i < total_error:
                    sample_to_remove = i
                    weights = weights_i
                    total_error = total_error_i
            
            ref_samples.remove(sample_to_remove)
            
            ref_samples_seq.append(list(np.copy(ref_samples)))
            weights_seq.append(np.copy(weights))
            total_error_seq.append(np.copy(total_error))
            
        ref_samples_seq.reverse()
        weights_seq.reverse()
        total_error_seq.reverse()
                
        return(ref_samples_seq, weights_seq, total_error_seq)
        
    def fit(self, networks, method='convex'):
        if method=='standard': 
            ref_samples_seq, = self._forward_bsa(networks, k_max=min(networks.shape[0], 30))
            ref_samples_seq, weights_seq, total_error_seq = self._backward_bsa(networks, ref_samples_init=ref_samples_seq[-1])
            self.ref_samples = ref_samples_seq[self.k_ref-1]
            self.ref_networks = networks[self.ref_samples]
            self.weights = weights[self.k_ref-1]
            
        if method=='convex': 
            ref_samples_seq, = self._forward_convex_bsa(networks, k_max=min(networks.shape[0], 30))
            ref_samples_seq, weights_seq, total_error_seq = self._backward_convex_bsa(networks, ref_samples_init=ref_samples_seq[-1])
            self.ref_samples = ref_samples_seq[self.k_ref-1]
            self.ref_networks = networks[self.ref_samples]
            self.weights = weights[self.k_ref-1]
        
    def fit_transform(self, networks, method='convex'):
        if method=='standard': 
            ref_samples_seq, = self._forward_bsa(networks, k_max=min(networks.shape[0], 30))
            ref_samples_seq, weights_seq, total_error_seq = self._backward_bsa(networks, ref_samples_init=ref_samples_seq[-1])
            self.ref_samples = ref_samples_seq[k_ref-1]
            self.ref_networks = networks[self.ref_samples]
            weights = weights[self.k_ref-1]
            
        if method=='convex': 
            ref_samples_seq, = self._forward_convex_bsa(networks, k_max=min(networks.shape[0], 30))
            ref_samples_seq, weights_seq, total_error_seq = self._backward_convex_bsa(networks, ref_samples_init=ref_samples_seq[-1])
            self.ref_samples = ref_samples_seq[self.k_ref-1]
            self.ref_networks = networks[self.ref_samples]
            weights = weights[self.k_ref-1]

        return(weights)

    def transform(self, networks, method='convex'):
        weights = np.zeros((networks.shape[0], self.k_ref))
        
        for i, X in enumerate(networks):
            if method=='standard': weights_i, error = self.barycentric_projection(X, self.ref_networks)
            if method=='convex': weights_i, error = self.convex_barycentric_projection(X, self.ref_networks)
            weights[i] = weights_i
        
        return(weights)
        
    def _visualize_2d(self, networks, ref_networks, weights):
        """Perform sample-limited backward convex barycentric subspace analysis 
        of a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
            Dataset.
        ref_networks : array-like, shape=[k_ref, n_nodes, n_nodes]
            Reference networks found by BSA.  
        weights : array-like, shape=[n_samples, k_ref]
            Barycentric weights of the projection of the dataset onto the 
            subspace of the reference networks.
            
        Returns
        -------
        networks_transformed : array-like, shape=[n_samples, 2]
            2D visualization of the dataset.
        """
        if len(ref_networks) == 1: ref_transformed = np.zeros(2)
        
        elif len(ref_networks) == 2: 
            a = np.linalg.norm(np.linalg.eigvalsh(ref_networks[1]) - np.linalg.eigvalsh(ref_networks[0]))
            ref_transformed = np.array([[0., 0.], [a, 0.]])
        
        elif len(ref_networks) == 3: 
            spectra = np.linalg.eigvalsh(ref_networks)
            a = np.linalg.norm(spectra[2] - spectra[1])
            b = np.linalg.norm(spectra[2] - spectra[0])
            c = np.linalg.norm(spectra[1] - spectra[0])
            ref_transformed = np.array([[0., 0.], [c, 0.], [(b ** 2 + c ** 2 - a ** 2) / (2. * c), np.sqrt(b ** 2 - ((b ** 2 + c ** 2 - a ** 2) / (2. * c)) ** 2)]])
        
        else: 
            mds = MDS(n_components=2)
            ref_transformed = mds.fit_transform(np.linalg.eigvalsh(ref_networks))
          
        networks_transformed = np.array([w_i @ ref_transformed for w_i in weights])
        
        return(networks_transformed)
            
    def visualize_2d(self, networks, method='convex'):
        weights = self.transform(networks)
        networks_transformed = self._visualize_2d(networks, self.ref_networks, weights)
        
        return(networks_transformed)
        
    
