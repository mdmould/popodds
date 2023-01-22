# popodds
Simple package for Bayesian model comparison.

Given samples from a posterior distribution inferred under some default prior, compute the Bayes factor or odds in favour of a new prior model.

## Installation

`pip install popodds`

## Usage

The package consists of the `ModelComparison` class to compute Bayes factors, and a wrapper function `log_odds` for simplicity.

The computation only requires a few ingredients:
- `model` a new prior model or samples from it,
- `prior` the original parameter estimation prior or samples from it
- `samples` samples from a parameter estimation run.

Optional:
- `model_bounds` optional parameter bounds for the new prior model,
- `prior_bounds` optional parameter bounds for the original prior model,
- `prior_odds` optional odds between the priors, which defaults to unity,
- `second_model` option model to compute odds against instead of prior,
- `second_model_bounds` optional parameter bounds for the second model.
