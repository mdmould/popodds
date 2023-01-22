import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import logsumexp


def log_odds(
    model,
    prior,
    samples,
    model_bounds=None,
    prior_bounds=None,
    prior_odds=1,
    second_model=None,
    second_model_bounds=None,
    ):
    """log posterior odds between model and prior or two models.
    
    Arguments
    ---------
    model: callable or array-like (d, n) or (n,)
        New model prior.
        - If a callable it should return the log PDF of the model for an
          input array-like (d, k) of d-dimensional samples.
        - If an array-like (d, n) it should contain d-dimensional model
          samples. Univariate data can have shape (n,).

    prior: callable or array-like (d, m) or (m,)
        Parameter estimation prior.
        - If a callable it should return the log PDF of the original
          parameter estimation prior for an input array-like (d, k) of
          d-dimensional samples.
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
          
    second_model_bounds: None, bool, or array-like
    [optional, Default = None]
        Parameter bounds used for second_model KDE if not already a
        callable. Allowed values as for model_bounds.
        
    Returns
    -------
    float
        log Bayes factor (or log posterior odds if prior_odds != 1)
    """
    
    log_bayes_factor = ModelComparison(
        model, prior, samples, model_bounds, prior_bounds,
        )()
    
    if second_model is not None:
        log_bayes_factor -= ModelComparison(
            second_model, prior, samples, second_model_bounds, prior_bounds,
            )()
        
    return log_bayes_factor + np.log(prior_odds)


class ModelComparison:
    
    def __init__(
        self, model, prior, samples, model_bounds=None, prior_bounds=None,
        ):
        """Perform Bayesan model comparison on event posteriors.
        
        Arguments
        ---------
        model: callable or array-like (d, n) or (n,)
            New model prior.
            - If a callable it should return the log PDF of the model for an
              input array-like (d, k) of d-dimensional samples.
            - If an array-like (d, n) it should contain d-dimensional model
              samples. Univariate data can have shape (n,).

        prior: callable or array-like (d, m) or (m,)
            Parameter estimation prior.
            - If a callable it should return the log PDF of the original
              parameter estimation prior for an input array-like (d, k) of d-
              dimensional samples.
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
        """
        
        self.samples = np.atleast_2d(samples)
        self.n_dim, self.n_samples = self.samples.shape
        assert self.n_dim < self.n_samples
        
        self.model = self.process_dist(model, bounds=model_bounds)
        self.prior = self.process_dist(prior, bounds=prior_bounds)
        
        self.cache = None
        
    def __call__(self):
        
        return self.log_bayes_factor()
        
    def bayes_factor(self):
        
        return np.exp(self.log_bayes_factor())
        
    def log_bayes_factor(self):
        
        if self.cache is None:
            log_model = self.model(self.samples)
            log_prior = self.prior(self.samples)
            log_weights = log_model - log_prior
            self.cache = logsumexp(log_weights) - np.log(self.n_samples)

        return self.cache
        
    def process_dist(self, dist, bounds=None):
        
        if callable(dist):
            return dist
        
        dist = np.atleast_2d(dist)
        assert dist.shape[0] == self.n_dim
        assert dist.shape[0] < dist.shape[1]
        
        return KDE(dist, bounds=bounds).log_pdf


## TODO
## Make bounded KDE more efficient
## Only reflect fraction f * bw of samples across boundaries
class KDE:
    
    def __init__(self, samples, bandwidth='scott', bounds=None):
        """Construct a (bounded) Gaussian kernel density estimate.
        
        Arguments
        ---------
        samples: array-like (n,) or (d, n)
            Parameter samples used to fit the KDE.
            - Univariate samples can have shape (n,) or (1, n).
            - d-dimensional samples have shape (d, n)
            
        bandwidth: str or scalar [optional, default = 'scott']
            Bandwidth to use for the Gaussian kernels.
            - Can be a string for rules of thumb 'scott' or 'silverman'.
            - A scalar constant sets the bandwidth scaling manually.
            See scipy.stats.gaussian_kde.
            
        bounds: None, bool, or array-like [optional, default = None]
            Parameter boundaries to truncate the support. The inputs samples
            are masked according to these bounds before kernel intialization.
            - A single value applies to all parameter dimensions.
            - For univariate data an array-like (2,) is allowed.
            - For multivariate data an array-like with d rows is allowed, where
              each row is either a single value or array-like (2,).
            - In all cases a None or False indicates no bound, a True
              indicates the bound is estimated from samples, while a number
              gives the location of the bound.
        """
        
        samples = np.atleast_2d(samples)
        assert samples.shape[0] < samples.shape[1]
        self.n_dim = samples.shape[0]
        
        if type(bandwidth) is str:
            bandwidth = bandwidth.lower()
            assert bandwidth == 'scott' or bandwidth == 'silverman'
        else:
            bandwidth = float(bandwidth)

        self.reflect = False
        if bounds is not None and bounds is not False:
            self.reflect = True
            self.bounds = self.get_bounds(samples, bounds)
            in_bounds = self.mask_data(samples)
            samples = samples[:, in_bounds]
            samples, self.n_reflections = self.reflect_samples(samples)
            
        self.kde = gaussian_kde(samples, bw_method=bandwidth)
        
    def pdf(self, points):
        """Evaluate the probability density of the KDE.
        
        Arguments
        ---------
        points: array-like (m,) or (d, m)
            Parameter locations at which to evaluate the KDE.
            - Univarate data can have shape (m,) or (1, m)
            - d-dimensional data has shape (d, n)
            
        Returns
        -------
        array-like (m,)
            Probability density evaluations at points.
        """
        
        points = np.atleast_2d(points)
        pdf = self.kde.pdf(points)
        
        if self.reflect:
            pdf = pdf * (self.n_reflections + 1)
            in_bounds = self.mask_data(points)
            pdf[~in_bounds] = 0.0
            
        return pdf
    
    def log_pdf(self, points):
        """Evaluate the log probability density of the KDE.
        
        Arguments
        ---------
        points: array-like (m,) or (d, m)
            Parameter locations at which to evaluate the KDE.
            - Univarate data can have shape (m,) or (1, m)
            - d-dimensional data has shape (d, n)
            
        Returns
        -------
        array-like (m,)
            log probability density evaluations at points.
        """
        
        points = np.atleast_2d(points)
        log_pdf = self.kde.logpdf(points)
        
        if self.reflect:
            log_pdf = log_pdf + np.log(self.n_reflections + 1)
            in_bounds = self.mask_data(points)
            log_pdf[~in_bounds] = -np.inf
            
        return log_pdf
            
    def get_bounds(self, samples, bounds):
        
        if bounds is True:
            mins = np.min(samples, axis=1)
            maxs = np.max(samples, axis=1)
            _bounds = np.transpose([mins, maxs])
            
            return _bounds
        
        if self.n_dim == 1:
            assert len(bounds) == 1 or len(bounds) == 2
            if len(bounds) == 2:
                bounds = [bounds]
        else:
            assert len(bounds) == self.n_dim

        _bounds = np.zeros((self.n_dim, 2))
        for dim in range(self.n_dim):
            if bounds[dim] is None or bounds[dim] is False:
                _bounds[dim] = [-np.inf, np.inf]
            elif bounds[dim] is True:
                _bounds[dim] = [np.min(samples[dim]), np.max(samples[dim])]
            else:
                for pos in range(2):
                    if bounds[dim][pos] is None or bounds[dim][pos] is False:
                        _bounds[dim][pos] = [-np.inf, np.inf][pos]
                    elif bounds[dim][pos] is True:
                        _bounds[dim][pos] = [np.min, np.max][pos](samples[dim])
                    else:
                        _bounds[dim][pos] = float(bounds[dim][pos])
                        
        return _bounds
                
    def reflect_samples(self, samples):
                
        n_reflections = 0
        _samples = samples.copy()
        for dim in range(self.n_dim):
            for pos in range(2):
                if np.isfinite(self.bounds[dim][pos]):
                    mirror = _samples.copy()
                    mirror[dim] = 2 * self.bounds[dim][pos] - mirror[dim]
                    samples = np.concatenate((samples, mirror), axis=1)
                    n_reflections += 1
                        
        return samples, n_reflections
            
    def mask_data(self, data):
                        
        data = np.atleast_2d(data)
        above = np.all(data >= self.bounds[:, [0]], axis=0)
        below = np.all(data <= self.bounds[:, [1]], axis=0)
        
        return above * below
        
