"""
PL Modeling Program
Python file 1/3: InteractivePLFittingGUI.py (implements the GUI)
-> Python file 2/3: PLModeling.py (implements the PL emission models)
Python file 3/3: InterferenceFunction.py (implements the interference function models)

Author: Erik Spaans
Date: March 2021
"""
import scipy.special
import scipy.constants as const
from scipy.constants import pi, c, h
from scipy import integrate
import numpy as np
k_eV = const.value('Boltzmann constant in eV/K')
h_eV = const.value('Planck constant in eV/Hz')
try:
    import pickle5 as pickle
except:
    import pickle

# Set proper path for files when executable is run
from os import path
import sys
path_to_file = getattr(sys, '_MEIPASS', path.abspath(path.dirname(__file__)))


class PLModel:
    """
    Models the PL spectra according to three possible models:
    BGF (band gap fluctuations)
    EF (electrostatic fluctuations)
    UPF (unified potential fluctuations: band gap + electrostatic fluctuations)
    """
    def __init__(self, params, E=(np.linspace(0.4, 1.5, 1111)), model='BGF'):
        """
        Initialise the class.
        params =
        BGF: mean_E_g, beta, sigma_g, T
        EF: E_g, theta, gamma, dmu, T, a0d
        UPF: mean_E_g, beta, sigma_g, theta, gamma, T, a0d
        """
        # Define class variables
        # params:
        self.params = params
        self.E = E

        # Fix quasi-fermi level splitting for BGF model
        # (does not affect normalised PL spectra)
        self.BGF_mu0 = 0.5

        file = open(path.join(path_to_file, 'files', 'conv2Dlookup.pkl'), 'rb')
        self.conv_theta = pickle.load(file)
        file.close()

        # Initialise class outputs
        self.abs, self.emission = np.zeros(E.shape), np.zeros(E.shape)

        self.solveFunction = self.solveBGF
        self.fitFunction = self.emissionBGFFit
        self.updateModel(model)

        # Solve model
        self.solve()

    def updateModel(self, model):
        """Update the PL model and corresponding functions."""
        if model == 'BGF':
            self.solveFunction = self.solveBGF
            self.fitFunction = self.emissionBGFFit
        elif model == 'EF':
            self.solveFunction = self.solveEF
            self.fitFunction = self.emissionEFFit
        elif model == 'UPF':
            self.solveFunction = self.solveUPF
            self.fitFunction = self.emissionUPFFit

    def solve(self):
        """Solve for absorption and emission."""
        self.solveFunction()

    def absBGF(self):
        """Compute the absorptance of the BGF model."""
        return 0.5*scipy.special.erfc((self.params[0]-self.E)
                                      / (2**0.5*self.params[2]))

    def emissionBGF(self, E, mean_E_g, beta, sigma_g, T):
        """Compute the emission of the BGF model."""
        return (pi/(h**3*c**2)
                * scipy.special.erfc((mean_E_g-E+beta*sigma_g**2/(k_eV*T))
                / (2**0.5*sigma_g))*(E*const.e)**2
                * np.exp(-(E-self.BGF_mu0-beta*mean_E_g)/(k_eV*T)
                         + 0.5*(beta*sigma_g/(k_eV*T))**2))

    def emissionBGFFit(self, E, mean_E_g, beta, sigma_g, T):
        """Fit the BGF model."""
        PL = self.emissionBGF(E, mean_E_g, beta, sigma_g, T)
        return PL/np.max(PL)

    def solveBGF(self):
        """Solve the BGF model."""
        self.abs = self.absBGF()
        self.emission = self.emissionBGF(self.E, self.params[0],
                                         self.params[1], self.params[2],
                                         self.params[3])

    def convolutionIntegralEF(self, E, E_g, theta, gamma):
        """Compute the convolution integral from EF model.
        Note: a 2D-lookup table is used instead of this function
        to speed up computations (see convolutionIntegralEFLookUp)."""
        deltaE = E-E_g

        def convIntegral():
            return np.array([integrate.quad(convFunc, -np.inf, deltaE[idx],
                                            args=(deltaE[idx],))[0]
                             for idx in range(deltaE.size)])

        def convFunc(x, arg):
            return np.exp(-np.abs(x/gamma)**theta)*(arg-x)**.5

        return convIntegral()/(gamma*2*scipy.special.gamma(1+1/theta))

    def convolutionIntegralEFLookUp(self, E, E_g, theta, gamma):
        """Lookup convolution integral from EF model from table."""
        deltaE_gamma = (E-E_g)/gamma
        return gamma**0.5*self.conv_theta(deltaE_gamma, theta)

    def emissionEF(self, conv, E, dmu, T, a0d):
        """Compute the emission of the EF model."""
        return (2*pi/(h**3*c**2)*(E*const.e)**2/(np.exp((E-dmu)/(k_eV*T))-1)
                * (1-np.exp(-conv*a0d*(1-2/(np.exp((E-dmu)/(2*k_eV*T))+1)))))

    def emissionEFFit(self, E, E_g, theta, gamma, dmu, T, a0d):
        """Fit the EF model."""
        conv = self.convolutionIntegralEFLookUp(E, E_g, theta, gamma)
        PL = self.emissionEF(conv, E, dmu, T, a0d)
        return PL/np.max(PL)

    def solveEF(self):
        """Solve the EF model."""
        conv = (self.convolutionIntegralEFLookUp
                (self.E, self.params[0], self.params[1], self.params[2]))
        self.abs = 1-np.exp(-conv*self.params[5])
        self.emission = self.emissionEF(conv, self.E, self.params[3],
                                        self.params[4], self.params[5])

    def emissionUPF(self, energy, mean_E_g, beta, sigma_g, theta, gamma,
                    T, a0d):
        """Compute the emission of the UPF model."""
        def absorptance(E_1, E):
            conv = self.convolutionIntegralEFLookUp(E, E_1, theta, gamma)
            # the values in conv are flipped because the 2d-lookup table
            # sorts the output automatically
            return 1-np.exp(-np.flip(conv)*a0d)

        def integralUPF(E_g_loc, E):
            absorp_gauss = (absorptance(E_g_loc, E)/(sigma_g*(2*pi)**0.5)
                            * np.exp(-(E_g_loc-mean_E_g)**2/(2*sigma_g**2)))
            emission = absorp_gauss*np.exp(-(E-beta*E_g_loc)/(k_eV*T))
            abs_int = np.trapz(absorp_gauss, E_sample)
            em_int = E**2*np.trapz(emission, E_sample)
            return abs_int, em_int

        E_sample_num = 1500
        E_sample = np.linspace(0.5, 2.0, E_sample_num)
        emission = np.zeros(energy.shape)
        absorp = np.zeros(energy.shape)
        for idx, E in enumerate(energy):
            absorp[idx], emission[idx] = integralUPF(E_sample, E)
        return absorp, emission

    def emissionUPFFit(self, energy, mean_E_g, beta, sigma_g, theta, gamma,
                       T, a0d):
        """Fit the UPF model."""
        emission = self.emissionUPF(energy, mean_E_g, beta, sigma_g, theta,
                                    gamma, T, a0d)[1]
        return emission/np.max(emission)

    def solveUPF(self):
        """Solve the UPF model."""
        self.abs, self.emission = (self.emissionUPF
                                   (self.E, self.params[0],
                                    self.params[1], self.params[2],
                                    self.params[3], self.params[4],
                                    self.params[5], self.params[6]))


def wave2energy(PL):
    """Convert PL data from wavelength to energy units."""
    E_data = h_eV*const.c/PL.wavelength
    PL_data = PL.PL_corrected*PL.wavelength**2/(h_eV*const.c)
    return E_data, PL_data/np.max(PL_data)
