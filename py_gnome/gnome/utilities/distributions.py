#!/usr/bin/env python
'''
Classes that generate various types of probability distributions
'''

import numpy
np = numpy

from gnome.utilities.compute_fraction import fraction_below_d


class UniformDistribution(object):
    'Uniform Probability Distribution'
    def __init__(self, low=0., high=0.1):
        '''
        :param low: For the Uniform distribution, it is lower bound.
        :param high: For the Uniform distribution, it is upper bound.
        '''
        self.low = low
        self.high = high
        self._check_uniform_args()

    def _check_uniform_args(self):
        if None in (self.low, self.high):
            raise TypeError('Uniform probability distribution requires '
                            'low and high')

    def _uniform(self, np_array):
        np_array[:] = np.random.uniform(self.low, self.high, len(np_array))

    def set_values(self, np_array):
        self._uniform(np_array)


class NormalDistribution(object):
    'Normal Probability Distribution'
    def __init__(self, mean=0., sigma=0.1):
        '''
        :param mean: The mean of the normal distribution
        :param sigma: The standard deviation of normal distribution
        '''
        self.mean = mean
        self.sigma = sigma
        self._check_normal_args()

    def _check_normal_args(self):
        if None in (self.mean, self.sigma):
            raise TypeError('Normal probability distribution requires '
                            'mean and sigma')

    def _normal(self, np_array):
        np_array[:] = np.random.normal(self.mean, self.sigma, len(np_array))

    def set_values(self, np_array):
        self._normal(np_array)


class LogNormalDistribution(object):
    'Log Normal Probability Distribution'
    def __init__(self, mean=0., sigma=0.1):
        '''
        :param mean: The mean of the normal distribution
        :param sigma: The standard deviation of normal distribution
        '''
        self.mean = mean
        self.sigma = sigma
        self._check_lognormal_args()

    def _check_lognormal_args(self):
        if None in (self.mean, self.sigma):
            raise TypeError('Log Normal probability distribution requires '
                            'mean and sigma')

    def _lognormal(self, np_array):
        np_array[:] = np.random.lognormal(self.mean, self.sigma, len(np_array))

    def set_values(self, np_array):
        self._lognormal(np_array)


class WeibullDistribution(object):
    'Log Normal Probability Distribution'
    def __init__(self, alpha=None, lambda_=1.0, min_=None, max_=None):
        '''
        :param alpha: The shape parameter 'alpha' - labeled as 'a' in
                      numpy.random.weibull distribution
        :param lambda_: The scale parameter for the distribution - required for
                        2-parameter weibull distribution (Rosin-Rammler).
        '''
        self.alpha = alpha
        self.lambda_ = lambda_
        self.min_ = min_
        self.max_ = max_
        self._check_weibull_args()

    def _check_weibull_args(self):
        if self.alpha is None:
            raise TypeError('Weibull distribution requires alpha')

        if self.min_ is not None:
            if self.min_ < 0:
                raise ValueError('Weibull distribution requires minimum >= 0')

            if fraction_below_d(self.min_, self.alpha, self.lambda_) > 0.999:
                raise ValueError('Weibull distribution requires '
                                 'minimum < 99.9% of total distribution')

        if self.max_ is not None:
            if self.max_ <= 0:
                raise ValueError('Weibull distribution requires maximum > 0')

            if fraction_below_d(self.max_, self.alpha, self.lambda_) < 0.001:
                raise ValueError('Weibull distribution requires '
                                 'maximum > 0.1% of total distribution')

            if self.min_ is not None and self.max_ < self.min_:
                raise ValueError('Weibull distribution requires '
                                 'maximum > minimum')

            if self.max_ < 0.00005:
                raise ValueError('Weibull distribution requires '
                                 'maximum > .000025 (25 microns)')

    def _weibull(self, np_array):
        np_array[:] = self.lambda_ * np.random.weibull(self.alpha,
                                                       len(np_array))

        if self.min_ is not None and self.max_ is not None:
            for x in range(len(np_array)):
                while np_array[x] < self.min_ or np_array[x] > self.max_:
                    np_array[x] = self.lambda_ * np.random.weibull(self.alpha)
        elif self.min_ is not None:
            for x in range(len(np_array)):
                while np_array[x] < self.min_:
                    np_array[x] = self.lambda_ * np.random.weibull(self.alpha)
        elif self.max_ is not None:
            for x in range(len(np_array)):
                while np_array[x] > self.max_:
                    np_array[x] = self.lambda_ * np.random.weibull(self.alpha)

    def set_values(self, np_array):
        self._weibull(np_array)


if __name__ == '__main__':
    # generates TypeError
    #DistributionBase()

    UniformDistribution(low=0, high=0.1)

    NormalDistribution(mean=0, sigma=0.1)

    LogNormalDistribution(mean=0, sigma=0.1)

    WeibullDistribution(alpha=1.8, lambda_=0.000248, min_=None, max_=None)
