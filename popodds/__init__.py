import numpy as np
from scipy.stats import gaussian_kde
import kalepy


def log_odds(*args, prior_odds=1, **kwargs):
    """ Compute the log_10 posterior odds for a model over the original prior.
    
    Arguments
    ---------
    sim_samples: array-like (M,) or (D, M,)
        Samples from a simulated model.
        Can be one-dimensional for univariate data with M samples.
        Must have shape (D, M,) for D-dimensional data.

    pe_samples: array-like (N,) or (D, N,)
        Samples from a parameter estimation posterior.
        Can be one-dimensional for univariate data with N samples.
        Must have shape (D, N,) for D-dimensional data.

    pe_prior: callable, array-like (N,) or (D, K,)
        Parameter estimation prior.
        - If a callable it should return the correctly normalized prior
          density for an input array-like (D, N,).
        - If an array-like (N,) it should contain evaluations of the prior
          at pe_samples.
        - If an array-like (D, K,) it should contain prior samples that can
          be used to construct a density estimate.

    sim_bounds: None, bool, or array-like [optional, Default = None]
        Parameter bounds used for density estimate of simulation samples.
        - A single value applies to all parameter dimensions.
        - For univariate data an array-like (2,) is allowed.
        - For multivariate data an array-like with D rows is allowed, where
          each row is either a single value or array-like (2,).
        - In all cases a None or False indicates no bound(s), a True
          indicates the bound is estimated from sim_samples, while a number
          gives the location of the bound.
        See kalepy.KDE for more details.

    pe_bounds: None, bool, or array-like [optional, Default = None]
        Parameter bounds used if pe_prior requires density estimate.
        Allowed values as for sim_bounds.
        
    prior_odds: number [optional, Default = 1]
        Prior ratio for new model over original.
    
    """
        
    return ModelComparison(*args, **kwargs)() + np.log10(prior_odds)


class ModelComparison:
    """Perform Bayesan model comparison on event posteriors.
    
    """
    
    def __init__(
        self, sim_samples, pe_samples, pe_prior, sim_bounds=None,
        pe_bounds=None,
        ):
        """Initialize comparison class with priors and posteriors.
        
        Arguments
        ---------
        sim_samples: array-like (M,) or (D, M,)
            Samples from a simulated model.
            Can be one-dimensional for univariate data with M samples.
            Must have shape (D, M,) for D-dimensional data.
            
        pe_samples: array-like (N,) or (D, N,)
            Samples from a parameter estimation posterior.
            Can be one-dimensional for univariate data with N samples.
            Must have shape (D, N,) for D-dimensional data.
            
        pe_prior: callable, array-like (N,) or (D, K,)
            Parameter estimation prior.
            - If a callable it should return the correctly normalized prior
              density for an input array-like (D, N,).
            - If an array-like (N,) it should contain evaluations of the prior
              at pe_samples.
            - If an array-like (D, K,) it should contain prior samples that can
              be used to construct a density estimate.
            
        sim_bounds: None, bool, or array-like [optional, Default = None]
            Parameter bounds used for density estimate of simulation samples.
            - A single value applies to all parameter dimensions.
            - For univariate data an array-like (2,) is allowed.
            - For multivariate data an array-like with D rows is allowed, where
              each row is either a single value or array-like (2,).
            - In all cases a None or False indicates no bound(s), a True
              indicates the bound is estimated from sim_samples, while a number
              gives the location of the bound.
            See kalepy.KDE for more details.
            
        pe_bounds: None, bool, or array-like [optional, Default = None]
            Parameter bounds used if pe_prior requires density estimate.
            Allowed values as for sim_bounds.
        
        """
        
        # Sample shapes should be (number of dimensions, number of samples,)
        sim_samples = np.atleast_2d(sim_samples)
        self.pe_samples = np.atleast_2d(pe_samples)
        assert sim_samples.shape[0] < sim_samples.shape[1]
        assert self.pe_samples.shape[0] < self.pe_samples.shape[1]
        assert sim_samples.shape[0] == self.pe_samples.shape[0]
        n_dim, n_pe = self.pe_samples.shape
        
        # Density estimate for simulation samples to evaluate on pe_samples
        self.sim_prior = self._kde(sim_samples, bounds=sim_bounds)
        
        # Process pe_prior, which can be callable or array-like
        # If a callable it will be evaluated on pe_samples
        if callable(pe_prior):
            self.pe_prior = pe_prior
        # If an array-like it can be prior evaluations or prior samples
        else:
            pe_prior = np.atleast_1d(pe_prior)
            if pe_prior.ndim == 1:
                # Prior density evaluated on pe_samples
                assert pe_prior.size == n_pe
                self.pe_prior = lambda _: pe_prior
            elif pe_prior.ndim == 2:
                # Prior samples for density estimate to evaluate on pe_samples
                assert pe_prior.shape[0] < pe_prior.shape[1]
                assert pe_prior.shape[0] == n_dim
                self.pe_prior = self._kde(pe_prior, bounds=pe_bounds)
                
        self._cache_pdf = None
        self._cache_prior = None
        
    def __call__(self):
        """Compute log_10 Bayes factor between simulation and PE prior.
        
        Returns
        -------
        float
            log_10 Bayes factor
            
        """
        
        if self._cache_pdf is None:
            self._cache_pdf = self.sim_prior(self.pe_samples)
        if self._cache_prior is None:
            self._cache_prior = self.pe_prior(self.pe_samples)
        
        return self._log_bayes_factor(self._cache_pdf, self._cache_prior)
    
    def _kde(self, samples, points=None, bounds=None):
        """Construct (bounded) Gaussian kernel density estimate.
        
        Arguments
        ---------
        samples: array-like (n,) or (D, n,)
            Parameter samples used to fit the KDE to.
            Univariate samples can have shape (n,).
            D-dimensional samples must have shape (D, n,).
            
        points: array-like (m,) or (D, m,) [optional, Default = None]
            Parameter points at which to evaluate the KDE.
            Univariate data can have shape (n,).
            D-dimensional data must have shape (D, m,).
            
        bounds: None, bool, or array-like [optional, Default = None]
            Parameter bounds used to truncate the KDE.
            - A single value applies to all parameter dimensions.
            - For univariate data an array-like (2,) is allowed.
            - For multivariate data an array-like with D rows is allowed, where
              each row is either a single value or array-like (2,).
            - In all cases a None or False indicates no bound(s), a True
              indicates the bound is estimated from sim_samples, while a number
              gives the location of the bound.
            See kalepy.KDE for more details.  
        
        Returns
        -------
        callable or array-like
            Return a callable KDE if points is None, otherwise an array-like
            where the KDE has been evaluated at points.
        
        """
        
        if not bounds:
            kde = gaussian_kde(samples)
        else:
            kde = kalepy.KDE(samples, reflect=bounds).pdf
            
        if points is None:
            return kde
        return kde(points)
    
    def _bayes_factor(self, pdf, prior):
        """ Compute the Bayes factor.
        
        The integral is computed as a Monte Carlo sum over posterior samples.
        
        Arguments
        ---------
        pdf: array-like (n,)
            Density evaluations for the new model at n posterior samples.
            
        prior: array-like (n,)
            Density evaluations for the original prior at n posterior samples.
            
        Returns
        -------
        number
            Bayes factor for the new model over the original prior.
        
        """
        
        assert pdf.size == prior.size
        
        return np.sum(pdf / prior) / pdf.size
    
    def _log_bayes_factor(self, pdf, prior):
        """ Compute the log_10 Bayes factor.
        
        The integral is computed as a Monte Carlo sum over posterior samples.
        
        Arguments
        ---------
        pdf: array-like (n,)
            Density evaluations for the new model at n posterior samples.
            
        prior: array-like (n,)
            Density evaluations for the original prior at n posterior samples.
            
        Returns
        -------
        number
            log_10 Bayes factor for the new model over the original prior.
        
        """

        return np.log10(self._bayes_factor(pdf, prior))

