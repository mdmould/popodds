import numpy as np
from scipy.special import logsumexp
from kaydee import KDE


def log_odds(
    model,
    prior,
    samples,
    log=True,
    model_bounds=None,
    prior_bounds=None,
    model_kwargs={},
    prior_kwargs={},
    prior_odds=1,
    detectable=None,
    second_model=None,
    second_bounds=None,
    second_kwargs={},
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

    log: bool [optional, Default = True]
        Callable model and prior are log PDF (True) or PDF (False).

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

    model_kwargs: dict [optional, Default = {}]
        Keyword arguments for model function or constructed KDE.

    prior_kwargs: dict [optional, Default = {}]
        Keyword arguments for prior function or constructed KDE.

    prior_odds: number [optional, Default = 1]
        Ratio of model priors.
        - If second_model is None, this is the prior odds of model over
          prior.
        - Otherwise it is the prior odds of model over second_model.

    detectable: None or dict [optional, Default = None]
        Detectable sources to compute the population detection fraction from.
        - If None then the odds between astrophysical populations is returned.
        - Otherwise it must be a dict with 'prior' and 'samples' keys.
        - detectable['samples'] is an array-like (d, l) of detectable sources.
          Univariate data can hav shape (l,).
        - detectable['prior'] is the PDF of all injected sources. It is either
          a function that returns the PDF or an array-like (l,) of PDF
          evaluations on detectable['samples']; log PDF if log = True.

    second_model: callable or array-like (d, n) or (n,)
    [optional, Default = None]
        Model to compute odds/Bayes factor against instead of the prior.
        - If a callable it should return the log PDF of the model for an
          input array-like (d, k) of d-dimensional samples.
        - If an array-like (d, n) it should contain d-dimensional model
          samples. Univariate data can have shape (n,).

    second_bounds: None, bool, or array-like [optional, Default = None]
        Parameter bounds used for second_model KDE if not already a
        callable. Allowed values as for model_bounds.

    second_kwargs: dict [optional, Default = {}]
        Keyword arguments for second_model function or constructed KDE.

    Returns
    -------
    float
        log Bayes factor (or log posterior odds if prior_odds != 1)
    """

    mc = ModelComparison(
        model, prior, samples, log,
        model_bounds, prior_bounds,
        model_kwargs, prior_kwargs,
        )
    log_bayes_factor = mc()

    if second_model is not None:
        second_mc = ModelComparison(
            second_model, prior, samples, log,
            second_bounds, prior_bounds,
            second_kwargs, prior_kwargs,
            )
        log_bayes_factor -= second_mc()

    # Reuse the ModelComparison class to compute the detectable fraction
    # because the form of the integral is the same.
    # The total number of injections cancels out.
    if detectable is not None:
        assert sorted(list(detectable.keys())) == ['prior', 'samples']

        samples = np.atleast_2d(detectable['samples'])
        assert samples.shape[0] == mc.n_dim
        assert samples.shape[0] < samples.shape[1]

        if callable(detectable['prior']):
            prior = detectable['prior']
        else:
            assert len(np.shape(detectable['prior'])) == 1
            assert np.size(detectable['prior']) == samples.shape[1]
            prior = lambda _: np.array(detectable['prior'])

        log_bayes_factor -= ModelComparison(
            mc.model, prior, samples, log, model_kwargs=model_kwargs,
            )()

        log_bayes_factor += ModelComparison(
            mc.prior if second_model is None else second_mc.model,
            prior,
            samples,
            log,
            model_kwargs={} if second_model is None else second_kwargs,
            )()

    return log_bayes_factor + np.log(prior_odds)


class ModelComparison:

    def __init__(
        self,
        model,
        prior,
        samples,
        log=True,
        model_bounds=None,
        prior_bounds=None,
        model_kwargs={},
        prior_kwargs={},
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

        log: bool [optional, Default = True]
            Callable model and prior are log PDF (True) or PDF (False).

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

        model_kwargs: dict [optional, Default = {}]
            Keyword arguments for model function or constructed KDE.

        prior_kwargs: dict [optional, Default = {}]
            Keyword arguments for prior function or constructed KDE.
        """

        self.samples = np.atleast_2d(samples)
        self.n_dim, self.n_samples = self.samples.shape
        assert self.n_dim < self.n_samples

        self.log = log

        self.model = self.process_dist(model, bounds=model_bounds)
        self.prior = self.process_dist(prior, bounds=prior_bounds)
        
        self.model_kwargs = model_kwargs
        self.prior_kwargs = prior_kwargs

        self.cache = None

    def __call__(self):

        return self.log_bayes_factor()

    def bayes_factor(self):

        return np.exp(self.log_bayes_factor())

    def log_bayes_factor(self):

        if self.cache is None:
            model = self.model(self.samples, **self.model_kwargs)
            prior = self.prior(self.samples, **self.prior_kwargs)

            if not self.log:
                model = np.log(model)
                prior = np.log(prior)

            self.cache = logsumexp(model - prior) - np.log(self.n_samples)

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


def heuristic_fraction(model, samples, quantile=0.9):
    """Compute number of model samples in posterior quantile.

    Arguments
    ---------
    model: array-like (d, n) or (n,)
        Model samples.
        - Must have shape (d, k) for d-dimensional data. Univariate data
          can have shape (k,)

    samples: array-like (d, k) or (k,)
        Parameter estimation samples.
        - Must have shape (d, k) for d-dimensional data. Univariate data
          can have shape (k,)

    quantile: float [optional, Default = 0.9]
        The symmetric central quantile level to define the posterior region.

    Returns
    -------
    float
        Heuristic fraction.
    """

    model = np.atleast_2d(model)
    samples = np.atleast_2d(samples)

    quantiles = 0.5 - quantile / 2, 0.5 + quantile / 2
    bounds = np.quantile(samples, quantiles, axis=1)[..., None]

    marginals = (bounds[0] < model) * (model < bounds[1])
    box = np.all(marginals, axis=0)

    return np.sum(box) / np.shape(model)[-1]


def relative_fraction(model, prior, samples, quantile=0.9):
    """Compute relative number of model/prior samples in posterior quantile.

    Arguments
    ---------
    model: array-like (d, n) or (n,)
        Model samples.
        - Must have shape (d, k) for d-dimensional data. Univariate data
          can have shape (k,)

    prior: array-like (d, n) or (n,)
        Prior samples.
        - Must have shape (d, k) for d-dimensional data. Univariate data
          can have shape (k,)

    samples: array-like (d, k) or (k,)
        Parameter estimation samples.
        - Must have shape (d, k) for d-dimensional data. Univariate data
          can have shape (k,)

    quantile: float [optional, Default = 0.9]
        The symmetric central quantile level to define the posterior region.

    Returns
    -------
    float
        Model heuristic fraction relative to prior heuristic fraction.
    """

    model = np.atleast_2d(model)
    prior = np.atleast_2d(prior)
    samples = np.atleast_2d(samples)

    quantiles = 0.5 - quantile / 2, 0.5 + quantile / 2
    bounds = np.quantile(samples, quantiles, axis=1)[..., None]

    box_model = np.all((bounds[0] < model) * (model < bounds[1]), axis=0)
    box_prior = np.all((bounds[0] < prior) * (prior < bounds[1]), axis=0)

    return (
        (np.sum(box_model) / np.shape(model)[-1]) /
        (np.sum(box_prior) / np.shape(prior)[-1])
        )

