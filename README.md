# popodds
Simple package for Bayesian model comparison.

Given samples from a posterior distribution inferred under some default prior, compute the Bayes factor or odds in favour of a new prior model.

## Usage

The package consists of the `ModelComparison` class to compute Bayes factors, and a wrapper function `log_odds` for simplicity.

The computation only requires a few ingredients:
- `model` a new prior model, e.g., samples from a simulation,
- `pe_samples` samples from a Bayesian parameter estimation run,
- `pe_prior` a function, prior evaluations, or prior samples corresponding to the original parameter estimation prior,
- `model_bounds` optional parameter bounds for the new prior model,
- `pe_bounds` optional parameter bounds for the original prior model,
- `prior_odds` optional odds between the prior models, which defaults to unity.
