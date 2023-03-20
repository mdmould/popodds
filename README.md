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
- `model_bounds` parameter bounds for the new prior model,
- `prior_bounds` parameter bounds for the original prior model,
- `log` compute probability densities in log space,
- `prior_odds` odds between the prior models, which defaults to unity,
- `second_model` model to compute odds against instead of prior,
- `second_bounds` parameter bounds for the second model,
- `detectable` compare between detectable rather than intrinsic populations.
