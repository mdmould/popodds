import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from astropy.cosmology import Planck15
import tensorflow as tf


class PowerlawRedshift:

    def __init__(self, z_min=0, z_max=2.3, z_pow=0):
        
        self.z_min = z_min
        self.z_max = z_max
        self.z_pow = z_pow

        z = np.linspace(z_min, z_max, 1_000)
        self.dVdz = interp1d(z, Planck15.differential_comoving_volume(z).value)
        self.norm = quad(self.model, z_min, z_max)[0]

        self.max_prob = -minimize_scalar(
            lambda z: -self.prob(z), bounds=[z_min, z_max], method='bounded',
            ).fun

    def model(self, z):

        return self.dVdz(z) * (1 + z)**(self.z_pow - 1)

    def prob(self, z):
        
        return (self.z_min < z) * (z < self.z_max) * self.model(z) / self.norm

    def sample(self, n=1):
        
        zs = np.array([])
        n_left = n
        while n_left > 0:
            z = np.random.uniform(self.z_min, self.z_max, n_left)
            p = np.random.uniform(0, self.max_prob, n_left)
            z_add = z[p < self.prob(z)][:n_left]
            zs = np.concatenate((zs, z_add))
            n_left -= z_add.size
                
        return zs


def spherical_to_cartesian(r, theta, phi):
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z


def lookup_limits():

    limits = {
        'mtot': [2, 1000],
        'q': [0.1, 1],
        'z': [1e-4, 4],
        'chi1x': [-1, 1],
        'chi1y': [-1, 1],
        'chi1z': [-1, 1],
        'chi2x': [-1, 1],
        'chi2y': [-1, 1],
        'chi2z': [-1, 1],
        'iota': [0, np.pi],
        'ra': [-np.pi, np.pi],
        'dec': [-np.pi/2, np.pi/2],
        'psi': [0, np.pi]
        }

    return limits


class Detection:
    
    def __init__(self, filename):
        
        self.model = self.load_model(filename)
        self.limits = lookup_limits()
    
    def load_model(self, filename):            

        try:
            return tf.keras.models.load_model(filename)
        except:
            raise ValueError(
                f'{filename} not found - download a model from' \
                'https://github.com/dgerosa/pdetclassifier',
                )
    
    def rescale(self, x, var):
        
        x = np.array(x)
        lo = min(self.limits[var])
        hi = max(self.limits[var])
        
        return 1 - 2 * (x - lo) / (hi - lo)
    
    def inputs(self, binaries):
        
        return np.transpose(
            [self.rescale(binaries[var], var) for var in self.limits],
            )
    
    def predict(self, binaries):
        
        x = self.inputs(binaries)
        y = self.model(x, training=False)
        
        return np.squeeze(y > 0.5).astype(int)
    
    def sample(self, n):
        
        n = int(n)
        binaries = {}
        
        for var in 'ra', 'psi':
            binaries[var] = np.random.uniform(*self.limits[var], n)
            
        binaries['iota'] = np.arccos(np.random.uniform(-1, 1, n))
        binaries['dec'] = np.arccos(np.random.uniform(-1, 1, n)) - np.pi/2
        
        binaries['z'] = PowerlawRedshift().sample(n)
        
        for i in 1, 2:
            r = np.random.uniform(0, 1, n)
            theta = np.arccos(np.random.uniform(-1, 1, n))
            phi = np.random.uniform(-np.pi, np.pi, n)
            x, y, z = spherical_to_cartesian(r, theta, phi)
            binaries[f'chi{i}x'] = x
            binaries[f'chi{i}y'] = y
            binaries[f'chi{i}z'] = z
            
        return binaries
    
    def average(self, binaries, n_average):
        
        n = binaries.pop('n')
        samples = self.sample(n_average)
        
        for var in binaries:
            binaries[var] = np.concatenate(
                np.repeat(binaries[var][:, None], n_average, axis=1),
                ).astype(np.float32)
        for var in samples:
            samples[var] = np.concatenate(
                np.repeat(samples[var][None, :], n, axis=0),
                ).astype(np.float32)
            
        if 'mtot' not in binaries:
            binaries['mtot'] = (
                (binaries['m1'] + binaries['m2'])
                * (1 + samples['z'])
                )
        
        if 'q' not in binaries:
            binaries['q'] = binaries['m2'] / binaries['m1']
        
        for var in self.limits:
            if var not in binaries:
                binaries[var] = samples[var]
                
        p = self.predict(binaries).reshape(n, n_average)
        
        return p.mean(axis=1)

