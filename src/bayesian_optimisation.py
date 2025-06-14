import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf

def new_candidate(
    z_values: list[np.ndarray],
    maximisation_quantity: list[np.ndarray],
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    num_candidates: int = 3,
) -> np.ndarray:
    """
    Given the current z values and their corresponding entropy values,
    this function returns the new candidate z value to evaluate next.
    
    Args:
    z_values: list[float] - the current z values
    entropy_values: list[float] - the corresponding entropy values
    
    Returns:
    float - the new candidate z value to evaluate
    """
    # Create the training data
    train_X = torch.tensor(np.array(z_values), dtype=torch.float64)
    train_Y = torch.tensor(maximisation_quantity, dtype=torch.float64).unsqueeze(-1)
    
    dimension = train_X.shape[-1]
        
    # Create the model
    gp = SingleTaskGP(
        train_X=train_X, 
        train_Y=train_Y,
        input_transform=Normalize(d=dimension),
        outcome_transform=Standardize(m=1)
        )
    
    # Fit the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Create the acquisition function
    EI = qLogExpectedImprovement(gp, best_f=train_Y.max(),)

    # Optimize the acquisition function
    candidates, acq_value = optimize_acqf(
        acq_function=EI,
        bounds=torch.tensor(np.array([lower_bound, upper_bound])),
        q=num_candidates,
        num_restarts=5,
        raw_samples=20,
    )
    
    return candidates.cpu().detach().numpy()
    
    