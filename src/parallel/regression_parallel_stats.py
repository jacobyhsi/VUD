"""Joint-Gaussian approximation to continuous Parallel PP (natural-log units)."""

import numpy as np


PP_GAUSSIAN_ESTIMATOR = "mc_joint_gaussian_conditional_entropy_ddof1_v1"
DEFAULT_PP_COVARIANCE_FLOOR = 1e-6


def joint_gaussian_statistics(samples, covariance_floor=DEFAULT_PP_COVARIANCE_FLOOR):
    """Fit one joint to paired outputs; derive all PP quantities from that fit.

    The floor is an absolute covariance eigenvalue floor in output-squared
    units. Zero disables regularization and rejects singular fits. This is
    the analytic entropy of the fitted Gaussian, NOT the in-sample mean NLL
    (which differs when covariance is estimated with ddof=1).
    """
    pairs = np.asarray(samples, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 3:
        raise ValueError("Gaussian parallel_pp requires at least 3 paired (u,y) samples")
    if not np.isfinite(pairs).all():
        raise ValueError("Joint samples must be finite")
    if not np.isfinite(covariance_floor) or covariance_floor < 0:
        raise ValueError("PP covariance floor must be finite and nonnegative")
    mean = pairs.mean(axis=0)
    raw_covariance = np.cov(pairs, rowvar=False, ddof=1)
    if not np.isfinite(mean).all() or not np.isfinite(raw_covariance).all():
        raise ValueError("Joint Gaussian moments overflowed")
    eigenvalues, eigenvectors = np.linalg.eigh(raw_covariance)
    if covariance_floor == 0 and eigenvalues[0] <= 0:
        raise ValueError("Singular joint covariance: use a positive --pp_covariance_floor")
    fitted_eigenvalues = np.maximum(eigenvalues, covariance_floor)
    regularized = bool(np.any(eigenvalues < covariance_floor))
    covariance = (
        (eigenvectors * fitted_eigenvalues) @ eigenvectors.T
        if regularized else raw_covariance.copy()
    )
    var_u, var_y = float(covariance[0, 0]), float(covariance[1, 1])
    cov_uy = float(covariance[0, 1])
    slope = cov_uy / var_u
    conditional_variance = var_y - slope * cov_uy
    if not np.isfinite(conditional_variance) or conditional_variance <= 0:
        raise ValueError("Nonpositive conditional variance; increase --pp_covariance_floor")
    va = float(0.5 * (np.log(2 * np.pi * np.e) + np.log(conditional_variance)))
    return {
        "mean": mean.tolist(),
        "raw_covariance": raw_covariance.tolist(),
        "covariance": covariance.tolist(),
        "raw_eigenvalues": eigenvalues.tolist(),
        "regularized": regularized,
        "covariance_floor": float(covariance_floor),
        "conditional_mean_slope": float(slope),
        "conditional_mean_intercept": float(mean[1] - slope * mean[0]),
        "conditional_variance": float(conditional_variance),
        "correlation": float(np.clip(cov_uy / np.sqrt(var_u) / np.sqrt(var_y), -1, 1)),
        "Va": va,
        "n": len(pairs),
    }
