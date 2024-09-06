"""HyperbolicTSNE estimator.
"""
import numpy as np

from sklearn.base import BaseEstimator

from sklearn.utils import check_random_state
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.validation import check_non_negative

from . import hd_mat_ as hd_mat
from .initializations_ import initialization
from .optimizer_ import SequentialOptimizer


class HyperbolicTSNE(BaseEstimator):
    """
    Estimator that permits computing Hyperbolic Embeddings of high-dimensional
    data. 

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.
    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.
    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a tuple (D, V), where
        D is the distance matrix and V, the high-dimensional matrix could be None.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.
    knn_neighbors: int, optional
        The number of neighbors to use for computing the kNN D matrix.
        This value might be overridden by certain methods like hd_method=vdm2008,
        which has perplexity.
    hd_method : str, optional (default: ("vdm2008", None)))
        Defines method used to compute the high-dimensional matrix.
        The default `perplexity_based`, uses the method by van der Maaten and Hinton.
        To see the available methods and their parameters check hd_mat_.available_methods
    hd_params: dict, optional
        Dict of custom parameters for the selected hd_method. Only params that are
        are listed for the method in hd_mat_.available_methods are supported.
    opt_method: tuple, optional default: SequentialOptimizer
        Defines the method used for the embedding optimization. The method can be a string, which refers to a template
        (see available templates with available_opt_templates), or an object of the type BaseOptimizer and its
        parameters. By default, this method uses the "bh_tsne" template, which corresponds to the method implemented in
        the paper "" by van der Maaten.
    opt_params: dict, optional default: SequentialOptimizer.sequence_vdm2008()
        Parameters that will be used by the optimizer defined in opt_method.
        The default is the sequence defined in the paper by van der Maaten and Hinton.
    verbose : int, optional (default: 0)
        Verbosity level. It will get propagated down.
    random_state : int, RandomState instance, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term: `Glossary <random_state>`.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="parecomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.22
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    cost_function_ : float
        Cost function value after optimization.
    n_iter_ : int
        Number of iterations run.
    Examples
    --------
    >>> import numpy as np
    >>> from hyperbolicTSNE import HyperbolicTSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = HyperbolicTSNE().fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    """

    @_deprecate_positional_args
    def __init__(self, n_components=2, *, init="random",
                 metric="euclidean", knn_neighbors=100,
                 hd_method="vdm2008", hd_params=None,
                 opt_method=SequentialOptimizer, opt_params=SequentialOptimizer.sequence_poincare(),
                 verbose=0, random_state=None, n_jobs=None):
        self.n_components = n_components
        self.init = init
        self.metric = metric
        self.knn_neighbors = knn_neighbors

        self.hd_method = hd_method
        self.hd_params = hd_params

        self.optimizer = None
        self.opt_method = opt_method
        self.opt_params = opt_params

        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.embedding_ = None
        self.cf = None                      # Cost function value
        self.runtime = 0.0                  # Runtime of embedding process
        self.its = 0                        # Number of iterations optimizer takes 

    def _fit(self, X):
        """Checks whether everything is in order to run the optimizer. Taking into account 
        self.hd_method, self.hd_params, self.opt_method, self.opt_params. 

        Parameters
        ----------
        X : array of shape (n_samples, n_features) or tuple with arrays (D, V)
        ((n_samples, n_samples),(n_samples, n_samples))
            If X is provided, D and V will be computed from it using the specified
            metric in hd_method. If a tuple is sent, the user is in charge of making
            sure that D is the distance matrix and V is the high-dimensional methods
            (e.g. P for tSNE).

        Returns
        -------
        X_new: array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """

        # Initialization and checks
        random_state = check_random_state(self.random_state)
        X_is_iterable = type(X) is tuple or type(X) is list

        if not X_is_iterable:
            if self.metric == "precomputed":
                raise TypeError(
                    "`metric` is precomputed, so X should be the tuple (D, V) with at least the distance matrix D")
            D = None
            V = None
            X = self._validate_data(X, accept_sparse=['csr'],
                                    ensure_min_samples=2,
                                    dtype=[np.float32, np.float64])
        else:
            if self.metric != "precomputed":
                raise ValueError(
                    "`metric` should be precomputed given that D is already given")
            if self.verbose > 0:
                print(
                    "[HyperbolicTSNE] Received iterable as input. It should have len=2 and contain (D=None, V=None)")
            if len(X) != 2:
                raise ValueError(
                    "X is an iterable, but does not have len=2, aborting")
            D, V = X

        # Checking hd_mat params
        if X_is_iterable:
            if isinstance(self.init, str) and self.init == "pca":
                raise ValueError(
                    "The parameter init=\"pca\" cannot be used with metric=\"precomputed\".")

            X = None
            if D is None and V is None:
                raise ValueError("[HyperbolicTSNE] Both D and V matrices cannot be None")
            else:
                if D is not None:
                    if D.shape[0] != D.shape[1]:
                        raise ValueError(
                            "[HyperbolicTSNE] D should be a square distance matrix")
                    check_non_negative(D, "HyperbolicTSNE.fit(). With hd_params.metric='precomputed', X[0] "
                                          "should contain positive distances.")
                    D = self._validate_data(D, accept_sparse=['csr'],
                                            ensure_min_samples=2,
                                            dtype=[np.float32, np.float64])
                else:
                    print(
                        "[HyperbolicTSNE] Warning: D matrix is None, make sure this is supported by the optimizer")
                if V is not None:
                    if V.shape[0] != V.shape[1]:
                        raise ValueError(
                            "[HyperbolicTSNE] V should be a square distance matrix")
                    check_non_negative(V, "HyperbolicTSNE.fit(). With hd_params.metric='precomputed', X[1] "
                                          "should contain positive distances.")
                    V = self._validate_data(V, accept_sparse=['csr'],
                                            ensure_min_samples=2,
                                            dtype=[np.float32, np.float64])
                else:
                    print(
                        "[HyperbolicTSNE] Warning: V matrix is None, make sure this is supported by the optimizer")


        # Compute high dim. affinity matrix
        D, V = hd_mat.hd_matrix(X=X, D=D, V=V,
                                metric=self.metric, n_neighbors=self.knn_neighbors,
                                hd_method=self.hd_method, hd_params=self.hd_params,
                                verbose=self.verbose)
        n_samples = V.shape[0]

        # Checking init
        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        else:
            X_embedded = initialization(
                n_samples, self.n_components, X, self.init, random_state=random_state)

        # Checking opt params
        self.opt_params["D"] = D  # Some optimizers need this

        # Initialize optimizer with correct parameters (set before we construct this class)
        # NOTE: The instance of the class (in self.opt_method) is explicitly created here
        self.optimizer = self.opt_method(
            Y0=X_embedded, V=V, n_components=self.n_components, other_params=self.opt_params, verbose=self.verbose)
        
        # Compute embeddings
        X_embedded, cf, runtime, its = self.optimizer.run()

        return X_embedded, cf, runtime, its

    def fit_transform(self, X, Y=None):
        """Fit X into an embedded space and return that transformed output.
        Parameters
        ----------
        X : array of shape (n_samples, n_features) or tuple with arrays (D, V)
        ((n_samples, n_samples),(n_samples, n_samples))
            If X is provided, D and V will be computed from it using the specified
            metric in hd_method. If a tuple is sent, the user is in charge of making
            sure that D is the distance matrix and V is the high-dimensional methods
            (e.g. P for tSNE).
        Y : Ignored

        Returns
        -------
        X_embedded : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        X_embedded, cf, runtime, its = self._fit(X)
        self.embedding_ = X_embedded
        self.cf = cf 
        self.runtime = runtime 
        self.its = its

        return self.embedding_

    def fit(self, X, Y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        Y : Ignored
        """
        self.fit_transform(X)
        return self
