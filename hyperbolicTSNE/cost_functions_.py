""" Implementation of Hyperbolic Kullback-Leibler Divergence

The cost function class has two main methods:
 - obj:     Objective function giving the cost function's value.
 - grad:    Gradient of the cost function.
"""

import ctypes

import numpy as np

from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

from hyperbolicTSNE.hyperbolic_barnes_hut.tsne import gradient, global_hsne_gradient, gaussian_gradient

from numpy import linalg as LA

MACHINE_EPSILON = np.finfo(np.double).eps
COMPUTE_EXACT_GRADIENT = False

def check_params(params):
    """Checks params dict includes supported key-values.
    Raises an exception if this is not the case.

    Parameters
    ----------
    params : _type_
        Cost function params in key-value format.
    """
    if "method" not in params or "params" not in params:
        raise ValueError("`other_params` should include a method string `method`, its params `params`.")
    if not isinstance(params["method"], str):
        raise TypeError("`method` of cost function should be a string.")
    if params["params"] is not None and not isinstance(params["params"], dict):
        raise TypeError("`params` should be either None or a dict with the appropriate setup parameters")

    general_params = ["num_threads", "verbose", "degrees_of_freedom", "calc_both", "area_split", "grad_fix"]
    if params["method"] == "exact":
        all_params = general_params + ["skip_num_points"]
    elif params["method"] == "barnes-hut":
        all_params = general_params + ["angle"]
    else:
        raise ValueError("HyperbolicKL method is not a valid one (available methods are `exact` and `barnes-hut`)")

    for p in params["params"]:
        if p not in all_params:
            raise ValueError(f"{p} is not in the param set of the `{params['method']}` version of HyperbolicKL.")
    for p in all_params:
        if p not in params["params"]:
            raise ValueError(
                f"{p} params is necessary for the `{params['method']}` version of HyperbolicKL. Please set a value or "
                f"use a preset."
            )


class HyperbolicKL:
    """
    Hyperbolic Kullback-Leibler Divergence cost function.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.
    other_params : dict
        Cost function params in key-value format.
    """
    def __init__(self, *, n_components, other_params=None):
        if other_params is None:
            raise ValueError(
                "No `other_params` specified for HyperbolicKL, please add your params or select one of the presets."
            )
        self.n_components = n_components
        self.params = other_params
        # Print whether were using the correct gradient or not
        # print("Grad Fix: ", self.params["params"]["grad_fix"])          # TODO: Eventually remove
        self.results = []

    @classmethod
    def class_str(cls):
        return f"HyperbolicKL"

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, other_params):
        try:
            default_params = self._params
        except AttributeError:
            default_params = {}
        if not isinstance(other_params, dict):
            raise Exception("`other_params` should be a dict ... initializing with default params")
        else:
            for k, v in other_params.items():
                default_params[k] = v
            check_params(default_params)
            self._params = default_params

    #######################
    # Parameter templates #
    #######################

    @classmethod
    def exact_tsne(cls):
        """Parameter preset for the exact Hyperbolic tSNE cost function.

        Returns
        -------
        dict
            Cost function params in key-value format.
        """
        return {
            "method": "exact",
            "params": {
                "degrees_of_freedom": 1,
                "skip_num_points": 0,
                "num_threads": _openmp_effective_n_threads(),
                "verbose": False
            }
        }

    @classmethod
    def bh_tsne(cls, angle=0.5):
        """Parameter preset for the accelerated Hyperbolic tSNE cost function.

        Parameters
        ----------
        angle : float, optional
            Degree of the approximation, by default 0.5

        Returns
        -------
        dict
            Cost function params in key-value format.
        """
        return {
            "method": "barnes-hut",
            "params": {"angle": angle, "degrees_of_freedom": 1, "num_threads": _openmp_effective_n_threads(),
                       "verbose": False}
        }

    #########################
    # User-facing functions #
    #########################

    def obj(self, Y, *, V, grad_fix):
        """Calculates the Hyperbolic KL Divergence of a given embedding.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).

        Returns
        -------
        float
            KL Divergence value.
        """
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            raise NotImplementedError("Exact obj not implemented. Use obj_grad to get exact cost function value.")
        elif self.params["method"] == "barnes-hut":
            obj, _ = self._obj_bh(Y, V, n_samples, grad_fix)
            return obj

    def grad(self, Y, *, V, grad_fix):
        """Calculates the gradient of the Hyperbolic KL Divergence of 
        a given embedding.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            return self._grad_exact(Y, V, n_samples, grad_fix)
        elif self.params["method"] == "barnes-hut":
            _, grad = self._grad_bh(Y, V, n_samples, grad_fix)
            return grad

    def obj_grad(self, Y, *, V, grad_fix):
        """Calculates the Hyperbolic KL Divergence and its gradient 
        of a given embedding.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).

        Returns
        -------
        float
            KL Divergence value.
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            obj, grad = self._grad_exact(Y, V, n_samples, grad_fix)
            return obj, grad
        elif self.params["method"] == "barnes-hut":
            obj, grad = self._obj_bh(Y, V, n_samples, grad_fix)
            return obj, grad

    ##########################
    # Main private functions #
    ##########################

    def _obj_exact(self, Y, V, n_samples):
        """Exact computation of the KL Divergence.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.
        """
        pass

    def _grad_exact(self, Y, V, n_samples, grad_fix, save_timings=True):
        """Exact computation of the KL Divergence gradient.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.
        save_timings : bool, optional
            If True, saves per iteration times, by default True.

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)         # shape(n_samples, n_components) - (n_samples, 2) usually
        timings = np.zeros(4, dtype=ctypes.c_float)
        error = gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            0.5,
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=True,
            grad_fix=grad_fix
        )

        grad = grad.ravel()
        grad *= 4

        if save_timings:
            self.results.append(timings)

        return error, grad

    def _obj_bh(self, Y, V, n_samples, grad_fix):
        """Approximate computation of the KL Divergence.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.

        Returns
        -------
        float
            KL Divergence value.
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        return self._grad_bh(Y, V, n_samples, grad_fix)

    def _grad_bh(self, Y, V, n_samples, grad_fix, save_timings=True):
        """Approximate computation of the KL Divergence gradient.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.
        save_timings : bool, optional
            If True, saves per iteration times, by default True.

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)
        timings = np.zeros(4, dtype=ctypes.c_float)
        error = gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            self.params["params"]["angle"],
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=False,
            area_split=self.params["params"]["area_split"],
            grad_fix=grad_fix
        )

        grad = grad.ravel()
        grad *= 4

        if save_timings:
            self.results.append(timings)

        return error, grad

















class CoSNE(HyperbolicKL):
    """
    Co-SNE Cost function
    Is HyperoblicKL + global_term. The global_term is = (1/n) * sum_i ([||x_i||^2 - ||y_i||^2])^2
    Which describes the difference of norms for the data, and the embedding. The idea is that this
    captures global structure to some degree.

    NOTE/TODO: Does it actually make sense to have a loss with this global_term that restricts the norms?
               y_i in D and x_i in R^n, which are not the same type of space at all. This norm comparison
               thus probably doesn't make sense?
    """
    def __init__(self, *, n_components, other_params=None):
        super().__init__(n_components=n_components, other_params=other_params)

    @classmethod
    def class_str(cls):
        return f"CoSNE"

    #########################
    # User-facing functions #
    #########################
    def obj(self, Y, *, V, x_norm, lambda_1, lambda_2, n_samples):
        obj, _ = self._obj_grad_bh(Y, V, x_norm, lambda_1, lambda_2, n_samples)
        return obj

    def grad(self, Y, *, V, x_norm, lambda_1, lambda_2, n_samples):
        _, grad = self._obj_grad_bh(Y, V, x_norm, lambda_1, lambda_2, n_samples)
        return grad

    def obj_grad(self, Y, *, V, x_norm, lambda_1, lambda_2, n_samples):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":       # NOTE: Requires a change in the tsne.pyx file. Doesnt do anything here...
            obj, grad = self._obj_grad_bh(Y, V, x_norm, lambda_1, lambda_2, n_samples)
            return obj, grad
        
        elif self.params["method"] == "barnes-hut":
            obj, grad = self._obj_grad_bh(Y, V, x_norm, lambda_1, lambda_2, n_samples)

            return obj, grad
    

    def _obj_grad_bh(self, Y, V, x_norm, lambda_1, lambda_2, n_samples, save_timings=True):
        """
            Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        n_samples : _type_
            Number of samples in the embedding.
        save_timings : bool, optional
            If True, saves per iteration times, by default True.

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        # Gradient is regular grad + extra term

        # Regular grad part
        error1, grad1 = super().obj_grad(Y, V=V)
        #print(grad1.shape, Y.shape, x_norm.shape)

        # Exta term part
        # error = sum_i -4 * (||x_i||^2 - ||y_i||^2) * y_i
        # originally Y is flatenned by .ravel()
        y = Y.reshape(n_samples, self.n_components)         # reshape to (n, 2)
        y_norm = (y * y).sum(axis=1)                        # y_norm has dim (n,)
        grad2 = -4 * ((x_norm - y_norm) * y.T).T            

        # grad is [g_1, ...,  g_n] and g_i = -4 * (||x_i||^2 - ||y_i||^2) * y_i
        error2 = grad2.sum()

        return error1 + error2, (lambda_1 * grad1) + (lambda_2 * grad2).ravel()
    

















class GlobalHSNE(HyperbolicKL):
    def __init__(self, *, n_components, other_params=None):
        super().__init__(n_components=n_components, other_params=other_params)

    @classmethod
    def class_str(cls):
        return f"GlobalHSNE"
    
    #########################
    # User-facing functions #
    #########################
    def obj(self, Y, *, V, P_hat, lbda):
        obj, _ = self._obj_grad_bh(Y, V)
        return obj

    def grad(self, Y, *, V, P_hat, lbda):
        _, grad = self._obj_grad_bh(Y, V)
        return grad

    def obj_grad(self, Y, *, V, P_hat, lbda):
        n_samples = V.shape[0]
        if self.params["method"] == "exact":       # NOTE: Requires a change in the tsne.pyx file. Doesnt do anything here...
            obj, grad = self._obj_grad_bh(Y, V, n_samples=n_samples)
            return obj, grad
        
        elif self.params["method"] == "barnes-hut":
            obj, grad = self._obj_grad_bh(Y, V, n_samples=n_samples)

            return obj, grad
    

    def _obj_grad_bh(self, Y, V, P_hat, lbda, n_samples, save_timings=True):
        """
            Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).
        P_hat : ndarray
            High dimensional global affinitiy matrix (as defined by global hsne)
        lbda : float
            Scalar term for additional gradient component
        save_timings : bool, optional
            If True, saves per iteration times, by default True.

        Returns
        -------
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        # Gradient is regular grad + extra term
        """
        def global_hsne_gradient(
             float[:] timings,                  # Array for trackig runtime
             double[:] val_P,                   # The high dimensional affinity data matrix (in CSR format)
             double[:, :] pos_output,           # Matrix containing embeddings (embedding)
             np.int64_t[:] neighbors,           # Column_indices of CSR matrix (for val_P)
             np.int64_t[:] indptr,              # Row_indices of CSR matrix (for val_P)
             np.int64_t lbda,                   # Scalar to multiply global gradient term by 
             double[:] global_P,                # Global affinity data matrix (in CSR format)
             np.int64_t[:] global_neighbours,   # Column indices of CSR matrix (for global_P)
             np.int64_t[:] global_indptr,       # Row_indices of CSR matrix (for global_P)
             double[:, :] forces,               # Matrix to store the forces in per element ij (initially all 0)
             float theta,                       # Threshold distance to use BH approximation for 
             int n_dimensions,                  # nr. of dimensions of high data (original high dim. data)
             int verbose,
             int dof=1,
             long skip_num_points=0,
             bint compute_error=1,
             int num_threads=1,
             bint exact=1,
             bint area_split=0,
             bint grad_fix=0):
        """
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)         # shape(n_samples, n_components) - (n_samples, 2) usually
        timings = np.zeros(4, dtype=ctypes.c_float)
        error = gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            0.5,
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=True,
            grad_fix=self.params["params"]["grad_fix"]
        )

        grad = grad.ravel()
        grad *= 4

        if save_timings:
            self.results.append(timings)
        return error, gradient
    
















##########################
# GAUSSIAN Cost Function #
##########################
class GaussianKL(HyperbolicKL):
    def __init__(self, *, n_components, other_params=None):
        super().__init__(n_components=n_components, other_params=other_params)

    @classmethod
    def class_str(cls):
        return f"GaussianKL"
    
    def obj_grad(self, Y, *, V, grad_fix, var):
        """Calculates the Hyperbolic KL Divergence and its gradient 
        of a given embedding.

        Parameters
        ----------
        Y : ndarray
            Flattened low dimensional embedding of length: n_samples x n_components.
        V : ndarray
            High-dimensional affinity matrix (P matrix in tSNE).

        Returns
        -------
        float
            KL Divergence value.
        ndarray
            Array (n_samples x n_components) with KL Divergence gradient values.
        """
        n_samples = V.shape[0]
        if self.params["method"] == "exact":
            obj, grad = self._grad_exact(Y, V, n_samples, grad_fix, var)
            return obj, grad
        elif self.params["method"] == "barnes-hut":
            obj, grad = self._grad_bh(Y, V, n_samples, grad_fix, var)
            return obj, grad
        
        
    def _grad_exact(self, Y, V, n_samples, grad_fix, var, exact=True, save_timings=True):
        Y = Y.astype(ctypes.c_double, copy=False)
        Y = Y.reshape(n_samples, self.n_components)

        val_V = V.data
        neighbors = V.indices.astype(np.int64, copy=False)
        indptr = V.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y.shape, dtype=ctypes.c_double)         # shape(n_samples, n_components) - (n_samples, 2) usually
        timings = np.zeros(4, dtype=ctypes.c_float)

        error = gaussian_gradient(
            timings,
            val_V, Y, neighbors, indptr, grad,
            0.5,
            self.n_components,
            self.params["params"]["verbose"],
            dof=self.params["params"]["degrees_of_freedom"],
            compute_error=True,
            num_threads=self.params["params"]["num_threads"],
            exact=exact,
            grad_fix=grad_fix,
            var=var
        )

        grad = grad.ravel()
        grad *= 4

        if save_timings:
            self.results.append(timings)

        return error, grad


    def _grad_bh(self, Y, V, n_samples, grad_fix, var, save_timings=True):
        return self._grad_exact(Y, V, n_samples, grad_fix, var, exact=False)