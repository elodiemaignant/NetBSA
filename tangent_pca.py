import numpy as np
from itertools import permutations
from sklearn.decomposition import PCA
from scipy.optimize import quadratic_assignment, minimize

class NetPCA:
    """Class for tangent principal component analysis (tangent PCA) of 
    network-valued data.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
        
    Attributes
    ----------

    n_nodes : int
        Number of nodes.
    n_samples : int
        Number of samples.
        
    References
    ----------
    .. [GSS2021] Guo, X., Srivastava, A. & Sarkar, S. (2021). "A quotient 
    space formulation for generative statistical analysis of graphical 
    data". JMIV 63, 735–752
    https://doi.org/10.1007/s10851-021-01027-1
    """
    
    def __init__(self, n_nodes):
        self.n_nodes=n_nodes

    def _log(self, network_1, network_2):
        """Compute the inverse of the exponential map from a first unlabeled 
        network to a second by solving the graph matching problem exactly.
        
        Parameters
        ----------
        network_1 : array-like, shape=[n_nodes, n_nodes]
        network_2 : array-like, shape=[n_nodes, n_nodes]
        
        Returns
        -------
        dist : float
            Inverse exponential map from network_1 to network_2.
        """
        permutation_mat = [] 
        S = permutations(np.arange(self.n_nodes)) 
        
        for s in S: 
            P = np.zeros_like(network_1)
            for (i, j) in enumerate(s): 
                P[i, j] = 1. 
            permutation_mat.append(P)
        
        i = np.argmin(np.array([np.sum((P @ network_2 @ np.transpose(P) - network_1) ** 2) for P in permutation_mat]))
        
        return(permutation_mat[i] @ network_2 @ permutation_mat[i].transpose() - network_1)

    def _log_fast(self, network_1, network_2):
        """Compute the inverse of the exponential map from a first unlabeled 
        network to a second using a fast but inexact algorithm for solving 
        the graph matching problem.
        
        Parameters
        ----------
        network_1 : array-like, shape=[n_nodes, n_nodes]
        network_2 : array-like, shape=[n_nodes, n_nodes]
        
        Returns
        -------
        dist : float
            Inverse exponential map from network_1 to network_2.
        """
        s = quadratic_assignment(network_1, network_2, method='faq', options={'maximize': True}).col_ind
        P = np.eye(self.n_nodes, dtype=int)[s]
        
        return(P @ network_2 @ P.transpose() - network_1)

    def _frechet_mean(self, networks):
        """Compute the frechet mean of a set of networks by solving the 
        graph matching problem exactly.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
        
        Returns
        -------
        frechet_mean : array-like, shape=[n_nodes, n_nodes]
            Frechet mean of the networks.
        jac : array-like, shape=[n_nodes, n_nodes]
            Norm of the MSE Jacobian after the last optimization step.
        success : bool
            Whether the optimization succeeded or not.
        """
        def mse(Y):
            s = 0.
            for X in networks: 
                log = self._log(Y.reshape(X.shape), X)
                s += np.sum(log ** 2)
            return(s / len(networks))

        def jac(Y):
            s = np.zeros(networks[0].shape)
            for X in networks: 
                log = self._log(Y.reshape(X.shape), X)
                s += log
            return(2. * s.flatten() / len(networks))

        Y0 = np.mean(networks, axis=0).flatten()
        # Y0 = networks[0].flatten()
        # Y0 = np.random.rand(self.n_nodes, self.n_nodes).flatten()
        res = minimize(mse, Y0, method='BFGS', tol=1E-6, options={'maxiter': 100})
        
        return(res.x.reshape(networks[0].shape), np.linalg.norm(jac(res.x.reshape(networks[0].shape))), res.success)

    def _frechet_mean_fast(self, networks):
        """Compute the frechet mean of a set of networks using a fast but 
        inexact algorithm for solving the graph matching problem.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
        
        Returns
        -------
        frechet_mean : array-like, shape=[n_nodes, n_nodes]
            Frechet mean of the networks.
        jac : array-like, shape=[n_nodes, n_nodes]
            Norm of the MSE Jacobian after the last optimization step.
        success : bool
            Whether the optimization succeeded or not.
        """
        def mse(Y):
            s = 0.
            for X in networks: 
                log = self._log_fast(self, Y.reshape(X.shape), X)
                s += np.sum(log ** 2)
            return(s / len(networks))

        def jac(Y):
            s = np.zeros(networks[0].shape)
            for X in networks: 
                log = self._log_fast(self, Y.reshape(X.shape), X)
                s += log
            return(2. * s.flatten() / len(networks))

        Y0 = np.mean(networks, axis=0).flatten()
        # Y0 = networks[0].flatten()
        # Y0 = np.random.rand(self.n_nodes, self.n_nodes).flatten()
        res = minimize(mse, Y0, method='BFGS', tol=1E-6, options={'maxiter': 100})
        
        return(res.x.reshape(networks[0].shape), np.linalg.norm(jac(res.x.reshape(networks[0].shape))), res.success)
        
    def tangent_pca(self, networks, n_components=2, fast=True):
        """Perform tangent PCA of a set of networks.
        
        Parameters
        ----------
        networks : array-like, shape=[n_samples, n_nodes, n_nodes]
        
        Returns
        -------
        networks_transformed : array-like, shape=[n_samples, n_components]
            Projection of the networks.
        components : array-like, shape=[n_components, n_nodes, n_nodes]
            Principal components.
        explained_variance : array-like, shape=[n_components]
            Explained variance.
        explained_variance_ratio : array-like, shape=[n_components]
            Explained variance ratios.
        """
        self.n_samples = networks.shape[0]
        
        if fast: mean, jac, success = self._frechet_mean_fast(networks)
        else: mean, jac, success = self._frechet_mean(networks)
        logs = np.array([self._log(mean, X).flatten() for X in networks])
        pca = PCA(n_components=n_components, svd_solver='full')
        
        return(pca.fit_transform(logs), pca.components_, pca.explained_variance_, pca.explained_variance_ratio_)
