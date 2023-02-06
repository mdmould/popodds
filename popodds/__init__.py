import numpy as np
from scipy.special import logsumexp
from kaydee import KDE


def log_odds(
    model,
    prior,
    samples,
    model_bounds=None,
    prior_bounds=None,
    log=True,
    prior_odds=1,
    second_model=None,
    second_model_bounds=None,
    ):
    """log posterior odds between model and prior or between two models.
    
    Arguments
    ---------
    model: callable or array-like (d, n) or (n,)
        New model prior.
        - If a callable it should return the PDF of the model for an input
          array-like (d, k) of d-dimensional samples. Should return the log
          PDF if log = True.
        - If an array-like (d, n) it should contain d-dimensional model
          samples. Univariate data can have shape (n,).

    prior: callable or array-like (d, m) or (m,)
        Parameter estimation prior.
        - If a callable it should return the PDF of the original parameter
          estimation prior for an input array-like (d, k) of d-dimensional
          samples. Should return the log PDF if log = True.
        - If an array-like (d, m) it should contain d-dimensional prior
          samples. Univariate data can have shape (m,).

    samples: array-like (d, k)
        Parameter estimation samples.
        - Must have shape (d, k) for d-dimensional data.

    model_bounds: None, bool, or array-like [optional, Default = None]
        Parameter bounds used for model KDE if not already a callable.
        - A single value applies to all parameter dimensions.
        - For univariate data an array-like (2,) is allowed.
        - For multivariate data an array-like with D rows is allowed,
          where each row is either a single value or array-like (2,).
        - In all cases a None or False indicates no bound(s), a True
          indicates the bound is estimated from samples, while a number
          gives the location of the bound.

    prior_bounds: None, bool, or array-like [optional, Default = None]
        Parameter bounds used for prior KDE if not already a callable.
        Allowed values as for model_bounds.
            
    log: bool [optional, Default = True]
        Callable model and prior are log PDF (True) or PDF (False).
        
    prior_odds: number [optional, Default = 1]
        Ratio of model priors.
        - If second_model is None, this is the prior odds of model over
          prior.
        - Otherwise it is the prior odds of model over second_model.
        
    second_model: callable or array-like (d, n) or (n,)
    [optional, Default = None]
        Model to compute odds/Bayes factor against instead of the prior.
        - If a callable it should return the log PDF of the model for an
          input array-like (d, k) of d-dimensional samples.
        - If an array-like (d, n) it should contain d-dimensional model
          samples. Univariate data can have shape (n,).
          
    second_bounds: None, bool, or array-like
    [optional, Default = None]
        Parameter bounds used for second_model KDE if not already a
        callable. Allowed values as for model_bounds.
        
    Returns
    -------
    float
        log Bayes factor (or log posterior odds if prior_odds != 1)
    """
    
    log_bayes_factor = ModelComparison(
        model, prior, samples, model_bounds, prior_bounds, log,
        )()
    
    if second_model is not None:
        log_bayes_factor -= ModelComparison(
            second_model, prior, samples, second_bounds, prior_bounds, log,
            )()
        
    return log_bayes_factor + np.log(prior_odds)


class ModelComparison:
    
    def __init__(
        self,
        model,
        prior,
        samples,
        model_bounds=None,
        prior_bounds=None,
        log=True,
        ):
        """Perform Bayesan model comparison on event posteriors.
        
        Arguments
        ---------
        model: callable or array-like (d, n) or (n,)
            New model prior.
            - If a callable it should return the PDF of the model for an input
              array-like (d, k) of d-dimensional samples. Should return the log
              PDF if log = True.
            - If an array-like (d, n) it should contain d-dimensional model
              samples. Univariate data can have shape (n,).

        prior: callable or array-like (d, m) or (m,)
            Parameter estimation prior.
            - If a callable it should return the PDF of the original parameter
              estimation prior for an input array-like (d, k) of d-dimensional
              samples. Should return the log PDF if log = True.
            - If an array-like (d, m) it should contain d-dimensional prior
              samples. Univariate data can have shape (m,).
            
        samples: array-like (d, k) or (k,)
            Parameter estimation samples.
            - Must have shape (d, k) for d-dimensional data. Univariate data
              can have shape (k,)
            
        model_bounds: None, bool, or array-like [optional, Default = None]
            Parameter bounds used for model KDE if not already a callable.
            - A single value applies to all parameter dimensions.
            - For univariate data an array-like (2,) is allowed.
            - For multivariate data an array-like with D rows is allowed, where
              each row is either a single value or array-like (2,).
            - In all cases a None or False indicates no bound(s), a True
              indicates the bound is estimated from samples, while a number
              gives the location of the bound.
            
        prior_bounds: None, bool, or array-like [optional, Default = None]
            Parameter bounds used for prior KDE if not already a callable.
            Allowed values as for model_bounds.
            
        log: bool [optional, Default = True]
            Callable model and prior are log PDF (True) or PDF (False).
        """
        
        self.samples = np.atleast_2d(samples)
        self.n_dim, self.n_samples = self.samples.shape
        assert self.n_dim < self.n_samples
        
        self.log = log
        
        self.model = self.process_dist(model, bounds=model_bounds)
        self.prior = self.process_dist(prior, bounds=prior_bounds)
        
        self.cache = None
        
    def __call__(self):
        
        return self.log_bayes_factor()
        
    def bayes_factor(self):
        
        return np.exp(self.log_bayes_factor())
        
    def log_bayes_factor(self):
        
        if self.cache is None:
            model = self.model(self.samples)
            prior = self.prior(self.samples)
            
            if self.log:
                self.cache = logsumexp(model - prior) - np.log(self.n_samples)
            else:
                self.cache = np.log(np.mean(model / prior))

        return self.cache
        
    def process_dist(self, dist, bounds=None):
        
        if callable(dist):
            return dist
        
        dist = np.atleast_2d(dist)
        assert dist.shape[0] == self.n_dim
        assert dist.shape[0] < dist.shape[1]
        
        kde = KDE(dist, bounds=bounds)
        
        if self.log:
            return kde.log_pdf
        return kde.pdf
