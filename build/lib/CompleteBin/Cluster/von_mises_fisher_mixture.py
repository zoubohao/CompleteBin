#######################################
## Copy from spectralcluster package ##
#######################################

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.special import iv  # modified Bessel function of first kind, I_v
from numpy import i0  # modified Bessel function of first kind order 0, I_0
from scipy.special import logsumexp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.extmath import squared_norm, stable_cumsum, row_norms
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from joblib import Parallel, delayed


MAX_CONTENTRATION = 1e10


def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if centers.shape[0] != n_centers:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of clusters {n_centers}.")
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of features of the data {X.shape[1]}.")


def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol



def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : int
        The number of seeds to choose

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers


def _init_centroids(X, n_clusters=8, init="k-means++", random_state=None,
                    x_squared_norms=None, init_size=None):
    """Compute the initial centroids

    Parameters
    ----------

    X : {ndarray, spare matrix} of shape (n_samples, n_features)
        The input samples.

    n_clusters : int, default=8
        number of centroids.

    init : {'k-means++', 'random', ndarray, callable}, default="k-means++"
        Method for initialization.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    x_squared_norms : ndarray of shape (n_samples,), default=None
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.

    Returns
    -------
    centers : array of shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    if init_size is not None and init_size < n_samples:
        if init_size < n_clusters:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, n_clusters),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * n_clusters
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < n_clusters:
        raise ValueError(
            "n_samples={} should be larger than n_clusters={}"
            .format(n_samples, n_clusters))

    if isinstance(init, str) and init == 'k-means++':
        centers = _k_init(X, n_clusters, random_state=random_state,
                          x_squared_norms=x_squared_norms)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(n_samples)[:n_clusters]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        # ensure that the centers have the same dtype as X
        # this is a requirement of fused types of cython
        centers = np.array(init, dtype=X.dtype)
    elif callable(init):
        centers = init(X, n_clusters, random_state=random_state)
        centers = np.asarray(centers, dtype=X.dtype)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(centers):
        centers = centers.toarray()

    _validate_center_shape(X, n_clusters, centers)
    return centers


def _inertia_from_labels(X, centers, labels):
    """Compute inertia with cosine distance using known labels.
    """
    n_examples, n_features = X.shape
    inertia = np.zeros((n_examples,))
    for ee in range(n_examples):
        inertia[ee] = 1 - X[ee, :].dot(centers[int(labels[ee]), :].T)

    return np.sum(inertia)


def _labels_inertia(X, centers):
    """Compute labels and inertia with cosine distance.
    """
    n_examples, n_features = X.shape
    n_clusters, n_features = centers.shape

    labels = np.zeros((n_examples,))
    inertia = np.zeros((n_examples,))

    for ee in range(n_examples):
        dists = np.zeros((n_clusters,))
        for cc in range(n_clusters):
            dists[cc] = 1 - X[ee, :].dot(centers[cc, :].T)

        labels[ee] = np.argmin(dists)
        inertia[ee] = dists[int(labels[ee])]

    return labels, np.sum(inertia)


def _vmf_log(X, kappa, mu):
    """Computs log(vMF(X, kappa, mu)) using built-in numpy/scipy Bessel
    approximations.

    Works well on small kappa and mu.
    """
    n_examples, n_features = X.shape
    return np.log(_vmf_normalize(kappa, n_features) * np.exp(kappa * X.dot(mu).T))


def _vmf_normalize(kappa, dim):
    """Compute normalization constant using built-in numpy/scipy Bessel
    approximations.

    Works well on small kappa and mu.
    """
    num = np.power(kappa, dim / 2. - 1.)

    if dim / 2. - 1. < 1e-15:
        denom = np.power(2. * np.pi, dim / 2.) * i0(kappa)
    else:
        denom = np.power(2. * np.pi, dim / 2.) * iv(dim / 2. - 1., kappa)

    if np.isinf(num):
        raise ValueError("VMF scaling numerator was inf.")

    if np.isinf(denom):
        raise ValueError("VMF scaling denominator was inf.")

    if np.abs(denom) < 1e-15:
        raise ValueError("VMF scaling denominator was 0.")

    return num / denom


def _log_H_asymptotic(nu, kappa):
    """Compute the Amos-type upper bound asymptotic approximation on H where
    log(H_\nu)(\kappa) = \int_0^\kappa R_\nu(t) dt.

    See "lH_asymptotic <-" in movMF.R and utility function implementation notes
    from https://cran.r-project.org/web/packages/movMF/index.html
    """
    beta = np.sqrt((nu + 0.5) ** 2)
    kappa_l = np.min([kappa, np.sqrt((3. * nu + 11. / 2.) * (nu + 3. / 2.))])
    return _S(kappa, nu + 0.5, beta) + (
        _S(kappa_l, nu, nu + 2.) - _S(kappa_l, nu + 0.5, beta)
    )


def _S(kappa, alpha, beta):
    """Compute the antiderivative of the Amos-type bound G on the modified
    Bessel function ratio.

    Note:  Handles scalar kappa, alpha, and beta only.

    See "S <-" in movMF.R and utility function implementation notes from
    https://cran.r-project.org/web/packages/movMF/index.html
    """
    kappa = 1. * np.abs(kappa)
    alpha = 1. * alpha
    beta = 1. * np.abs(beta)
    a_plus_b = alpha + beta
    u = np.sqrt(kappa ** 2 + beta ** 2)
    if alpha == 0:
        alpha_scale = 0
    else:
        alpha_scale = alpha * np.log((alpha + u) / a_plus_b)

    return u - beta - alpha_scale


def _vmf_log_asymptotic(X, kappa, mu):
    """Compute log(f(x|theta)) via Amos approximation

        log(f(x|theta)) = theta' x - log(H_{d/2-1})(\|theta\|)

    where theta = kappa * X, \|theta\| = kappa.

    Computing _vmf_log helps with numerical stability / loss of precision for
    for large values of kappa and n_features.

    See utility function implementation notes in movMF.R from
    https://cran.r-project.org/web/packages/movMF/index.html
    """
    n_examples, n_features = X.shape
    log_vfm = kappa * X.dot(mu).T + -_log_H_asymptotic(n_features / 2. - 1., kappa)

    return log_vfm


def _log_likelihood(X, centers, weights, concentrations):
    if len(np.shape(X)) != 2:
        X = X.reshape((1, len(X)))

    n_examples, n_features = np.shape(X)
    n_clusters, _ = centers.shape

    if n_features <= 50:  # works up to about 50 before numrically unstable
        vmf_f = _vmf_log
    else:
        vmf_f = _vmf_log_asymptotic

    f_log = np.zeros((n_clusters, n_examples))
    for cc in range(n_clusters):
        f_log[cc, :] = vmf_f(X, concentrations[cc], centers[cc, :])

    posterior = np.zeros((n_clusters, n_examples))
    weights_log = np.log(weights)
    posterior = np.tile(weights_log.T, (n_examples, 1)).T + f_log
    for ee in range(n_examples):
        posterior[:, ee] = np.exp(posterior[:, ee] - logsumexp(posterior[:, ee]))

    return posterior


def _init_unit_centers(X, n_clusters, random_state, init):
    """Initializes unit norm centers.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    init:  (string) one of
        k-means++ : uses sklearn k-means++ initialization algorithm
        spherical-k-means : use centroids from one pass of spherical k-means
        random : random unit norm vectors
        random-orthonormal : random orthonormal vectors
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    """
    n_examples, n_features = np.shape(X)
    if isinstance(init, np.ndarray):
        n_init_clusters, n_init_features = init.shape
        assert n_init_clusters == n_clusters
        assert n_init_features == n_features

        # ensure unit normed centers
        centers = init
        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers

    # elif init == "spherical-k-means":
    #     labels, inertia, centers, iters = spherical_kmeans._spherical_kmeans_single_lloyd(
    #         X, n_clusters, x_squared_norms=np.ones((n_examples,)), init="k-means++"
    #     )

        # return centers

    elif init == "random":
        centers = np.random.randn(n_clusters, n_features)
        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers

    elif init == "k-means++":
        centers = _init_centroids(
            X,
            n_clusters,
            "k-means++",
            random_state=random_state,
            x_squared_norms=np.ones((n_examples,)),
        )

        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers

    elif init == "random-orthonormal":
        centers = np.random.randn(n_clusters, n_features)
        q, r = np.linalg.qr(centers.T, mode="reduced")

        return q.T

    elif init == "random-class":
        centers = np.zeros((n_clusters, n_features))
        for cc in range(n_clusters):
            while np.linalg.norm(centers[cc, :]) == 0:
                labels = np.random.randint(0, n_clusters, n_examples)
                centers[cc, :] = X[labels == cc, :].sum(axis=0)

        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers


def _expectation(X, centers, weights, concentrations, posterior_type="soft"):
    """Compute the log-likelihood of each datapoint being in each cluster.

    Parameters
    ----------
    centers (mu) : array, [n_centers x n_features]
    weights (alpha) : array, [n_centers, ] (alpha)
    concentrations (kappa) : array, [n_centers, ]

    Returns
    ----------
    posterior : array, [n_centers, n_examples]
    """
    n_examples, n_features = np.shape(X)
    n_clusters, _ = centers.shape

    if n_features <= 50:  # works up to about 50 before numrically unstable
        vmf_f = _vmf_log
    else:
        vmf_f = _vmf_log_asymptotic

    f_log = np.zeros((n_clusters, n_examples))
    for cc in range(n_clusters):
        f_log[cc, :] = vmf_f(X, concentrations[cc], centers[cc, :])

    posterior = np.zeros((n_clusters, n_examples))
    if posterior_type == "soft":
        weights_log = np.log(weights)
        posterior = np.tile(weights_log.T, (n_examples, 1)).T + f_log
        for ee in range(n_examples):
            posterior[:, ee] = np.exp(posterior[:, ee] - logsumexp(posterior[:, ee]))

    elif posterior_type == "hard":
        weights_log = np.log(weights)
        weighted_f_log = np.tile(weights_log.T, (n_examples, 1)).T + f_log
        for ee in range(n_examples):
            posterior[np.argmax(weighted_f_log[:, ee]), ee] = 1.0

    return posterior


def _maximization(X, posterior, force_weights=None):
    """Estimate new centers, weights, and concentrations from

    Parameters
    ----------
    posterior : array, [n_centers, n_examples]
        The posterior matrix from the expectation step.

    force_weights : None or array, [n_centers, ]
        If None is passed, will estimate weights.
        If an array is passed, will use instead of estimating.

    Returns
    ----------
    centers (mu) : array, [n_centers x n_features]
    weights (alpha) : array, [n_centers, ] (alpha)
    concentrations (kappa) : array, [n_centers, ]
    """
    n_examples, n_features = X.shape
    n_clusters, n_examples = posterior.shape
    concentrations = np.zeros((n_clusters,))
    centers = np.zeros((n_clusters, n_features))
    if force_weights is None:
        weights = np.zeros((n_clusters,))

    for cc in range(n_clusters):
        # update weights (alpha)
        if force_weights is None:
            weights[cc] = np.mean(posterior[cc, :])
        else:
            weights = force_weights

        # update centers (mu)
        X_scaled = X.copy()
        if sp.issparse(X):
            X_scaled.data *= posterior[cc, :].repeat(np.diff(X_scaled.indptr))
        else:
            for ee in range(n_examples):
                X_scaled[ee, :] *= posterior[cc, ee]

        centers[cc, :] = X_scaled.sum(axis=0)

        # normalize centers
        center_norm = np.linalg.norm(centers[cc, :])
        if center_norm > 1e-8:
            centers[cc, :] = centers[cc, :] / center_norm

        # update concentration (kappa) [TODO: add other kappa approximations]
        rbar = center_norm / (n_examples * weights[cc])
        concentrations[cc] = rbar * n_features - np.power(rbar, 3.)
        if np.abs(rbar - 1.0) < 1e-10:
            concentrations[cc] = MAX_CONTENTRATION
        else:
            concentrations[cc] /= 1. - np.power(rbar, 2.)

        # let python know we can free this (good for large dense X)
        del X_scaled

    return centers, weights, concentrations


def _movMF(
    X,
    n_clusters,
    posterior_type="soft",
    force_weights=None,
    max_iter=300,
    verbose=False,
    init="random-class",
    random_state=None,
    tol=1e-6,
):
    """Mixture of von Mises Fisher clustering.

    Implements the algorithms (i) and (ii) from

      "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"
      by Banerjee, Dhillon, Ghosh, and Sra.

    TODO: Currently only supports Banerjee et al 2005 approximation of kappa,
          however, there are numerous other approximations see _update_params.

    Attribution
    ----------
    Approximation of log-vmf distribution function from movMF R-package.

    movMF: An R Package for Fitting Mixtures of von Mises-Fisher Distributions
    by Kurt Hornik, Bettina Grun, 2014

    Find more at:
      https://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
      https://cran.r-project.org/web/packages/movMF/index.html

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    posterior_type: 'soft' or 'hard'
        Type of posterior computed in exepectation step.
        See note about attribute: self.posterior_

    force_weights : None or array [n_clusters, ]
        If None, the algorithm will estimate the weights.
        If an array of weights, algorithm will estimate concentrations and
        centers with given weights.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init:  (string) one of
        random-class [default]: random class assignment & centroid computation
        k-means++ : uses sklearn k-means++ initialization algorithm
        spherical-k-means : use centroids from one pass of spherical k-means
        random : random unit norm vectors
        random-orthonormal : random orthonormal vectors
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-6
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    """
    random_state = check_random_state(random_state)
    n_examples, n_features = np.shape(X)

    # init centers (mus)
    centers = _init_unit_centers(X, n_clusters, random_state, init)

    # init weights (alphas)
    if force_weights is None:
        weights = np.ones((n_clusters,))
        weights = weights / np.sum(weights)
    else:
        weights = force_weights

    # init concentrations (kappas)
    concentrations = np.ones((n_clusters,))

    if verbose:
        print("Initialization complete")

    for iter in range(max_iter):
        centers_prev = centers.copy()

        # expectation step
        posterior = _expectation(
            X, centers, weights, concentrations, posterior_type=posterior_type
        )

        # maximization step
        centers, weights, concentrations = _maximization(
            X, posterior, force_weights=force_weights
        )

        # check convergence
        tolcheck = squared_norm(centers_prev - centers)
        if tolcheck <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e" % (iter, tolcheck, tol)
                )
            break

    # labels come for free via posterior
    labels = np.zeros((n_examples,))
    for ee in range(n_examples):
        labels[ee] = np.argmax(posterior[:, ee])

    inertia = _inertia_from_labels(X, centers, labels)

    return centers, weights, concentrations, posterior, labels, inertia


def movMF(
    X,
    n_clusters,
    posterior_type="soft",
    force_weights=None,
    n_init=10,
    n_jobs=1,
    max_iter=300,
    verbose=False,
    init="random-class",
    random_state=None,
    tol=1e-6,
    copy_x=True,
):
    """Wrapper for parallelization of _movMF and running n_init times.
    """
    if n_init <= 0:
        raise ValueError(
            "Invalid number of initializations."
            " n_init=%d must be bigger than zero." % n_init
        )
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError(
            "Number of iterations should be a positive number,"
            " got %d instead" % max_iter
        )

    best_inertia = np.infty
    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    if hasattr(init, "__array__"):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: "
                "performing only one init in k-means instead of n_init=%d" % n_init,
                RuntimeWarning,
                stacklevel=2,
            )
            n_init = 1

    # defaults
    best_centers = None
    best_labels = None
    best_weights = None
    best_concentrations = None
    best_posterior = None
    best_inertia = None

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # cluster on the sphere
            (centers, weights, concentrations, posterior, labels, inertia) = _movMF(
                X,
                n_clusters,
                posterior_type=posterior_type,
                force_weights=force_weights,
                max_iter=max_iter,
                verbose=verbose,
                init=init,
                random_state=random_state,
                tol=tol,
            )

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_weights = weights.copy()
                best_concentrations = concentrations.copy()
                best_posterior = posterior.copy()
                best_inertia = inertia
    else:
        # parallelisation of movMF runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_movMF)(
                X,
                n_clusters,
                posterior_type=posterior_type,
                force_weights=force_weights,
                max_iter=max_iter,
                verbose=verbose,
                init=init,
                random_state=random_state,
                tol=tol,
            )
            for seed in seeds
        )

        # Get results with the lowest inertia
        centers, weights, concentrations, posteriors, labels, inertia = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_concentrations = concentrations[best]
        best_posterior = posteriors[best]
        best_weights = weights[best]

    return (
        best_centers,
        best_labels,
        best_inertia,
        best_weights,
        best_concentrations,
        best_posterior,
    )


class VonMisesFisherMixture(BaseEstimator, ClusterMixin, TransformerMixin):
    """Estimator for Mixture of von Mises Fisher clustering on the unit sphere.

    Implements the algorithms (i) and (ii) from

      "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"
      by Banerjee, Dhillon, Ghosh, and Sra.

    TODO: Currently only supports Banerjee et al 2005 approximation of kappa,
          however, there are numerous other approximations see _update_params.

    Attribution
    ----------
    Approximation of log-vmf distribution function from movMF R-package.

    movMF: An R Package for Fitting Mixtures of von Mises-Fisher Distributions
    by Kurt Hornik, Bettina Grun, 2014

    Find more at:
      https://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
      https://cran.r-project.org/web/packages/movMF/index.html

    Basic sklearn scaffolding from sklearn.cluster.KMeans.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    posterior_type: 'soft' or 'hard'
        Type of posterior computed in exepectation step.
        See note about attribute: self.posterior_

    force_weights : None or array [n_clusters, ]
        If None, the algorithm will estimate the weights.
        If an array of weights, algorithm will estimate concentrations and
        centers with given weights.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init:  (string) one of
        random-class [default]: random class assignment & centroid computation
        k-means++ : uses sklearn k-means++ initialization algorithm
        random : random unit norm vectors
        random-orthonormal : random orthonormal vectors
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-6
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    normalize : boolean, default True
        Normalize the input to have unnit norm.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    weights_ : array, [n_clusters,]
        Weights of each cluster in vMF distribution (alpha).

    concentrations_ : array [n_clusters,]
        Concentration parameter for each cluster (kappa).
        Larger values correspond to more concentrated clusters.

    posterior_ : array, [n_clusters, n_examples]
        Each column corresponds to the posterio distribution for and example.

        If posterior_type='hard' is used, there will only be one non-zero per
        column, its index corresponding to the example's cluster label.

        If posterior_type='soft' is used, this matrix will be dense and the
        column values correspond to soft clustering weights.
    """

    def __init__(
        self,
        n_clusters=5,
        posterior_type="soft",
        force_weights=None,
        n_init=10,
        n_jobs=1,
        max_iter=300,
        verbose=False,
        init="random-class",
        random_state=None,
        tol=1e-6,
        copy_x=True,
        normalize=True,
    ):
        self.n_clusters = n_clusters
        self.posterior_type = posterior_type
        self.force_weights = force_weights
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.tol = tol
        self.copy_x = copy_x
        self.normalize = normalize

    def _check_force_weights(self):
        if self.force_weights is None:
            return

        if len(self.force_weights) != self.n_clusters:
            raise ValueError(
                (
                    "len(force_weights)={} but must equal "
                    "n_clusters={}".format(len(self.force_weights), self.n_clusters)
                )
            )

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "n_samples=%d should be >= n_clusters=%d"
                % (X.shape[0], self.n_clusters)
            )

        for ee in range(n_samples):
            if sp.issparse(X):
                n = sp.linalg.norm(X[ee, :])
            else:
                n = np.linalg.norm(X[ee, :])

            if np.abs(n - 1.) > 1e-4:
                raise ValueError("Data l2-norm must be 1, found {}".format(n))

        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse="csr", dtype=FLOAT_DTYPES, warn_on_dtype=True)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError(
                "Incorrect number of features. "
                "Got %d features, expected %d" % (n_features, expected_n_features)
            )

        for ee in range(n_samples):
            if sp.issparse(X):
                n = sp.linalg.norm(X[ee, :])
            else:
                n = np.linalg.norm(X[ee, :])

            if np.abs(n - 1.) > 1e-4:
                raise ValueError("Data l2-norm must be 1, found {}".format(n))

        return X

    def fit(self, X, y=None):
        """Compute mixture of von Mises Fisher clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        if self.normalize:
            X = normalize(X)

        self._check_force_weights()
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.weights_,
            self.concentrations_,
            self.posterior_,
        ) = movMF(
            X,
            self.n_clusters,
            posterior_type=self.posterior_type,
            force_weights=self.force_weights,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            verbose=self.verbose,
            init=self.init,
            random_state=random_state,
            tol=self.tol,
            copy_x=self.copy_x,
        )

        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        return self.fit(X)._transform(X)

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the cosine distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        if self.normalize:
            X = normalize(X)

        check_is_fitted(self, "cluster_centers_")
        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return cosine_distances(X, self.cluster_centers_)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Note:  Does not check that each point is on the sphere.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.normalize:
            X = normalize(X)

        check_is_fitted(self, "cluster_centers_")

        X = self._check_test_data(X)
        return _labels_inertia(X, self.cluster_centers_)[0]

    def score(self, X, y=None):
        """Inertia score (sum of all distances to closest cluster).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.

        Returns
        -------
        score : float
            Larger score is better.
        """
        if self.normalize:
            X = normalize(X)

        check_is_fitted(self, "cluster_centers_")
        X = self._check_test_data(X)
        return -_labels_inertia(X, self.cluster_centers_)[1]

    def log_likelihood(self, X):
        check_is_fitted(self, "cluster_centers_")

        return _log_likelihood(
            X, self.cluster_centers_, self.weights_, self.concentrations_
        )
