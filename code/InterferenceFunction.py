"""
PL Modeling Program
Python file 1/3: InteractivePLFittingGUI.py (implements the GUI)
Python file 2/3: PLModeling.py (implements the PL emission models)
-> Python file 3/3: InterferenceFunction.py (implements the interference function models)

Author: Erik Spaans
Date: March 2021
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal

# Set proper path for files when executable is run
from os import path
import sys
path_to_file = getattr(sys, '_MEIPASS', path.abspath(path.dirname(__file__)))


class IF:
    """
    Computes the interference function for a stack of three homogeneous
    layers (air - CIGS - Mo).
    """
    def __init__(self, d, wave_laser, dz=1e-9, theta_i=0, opt_data=None,
                 wavelengths=(1e-9*np.linspace(900, 1400, 501)),
                 luminescence='uniform', params=None,
                 rough=None, k_Mo=1):
        """
        Initialise the class.
        params =
        uniform: -
        delta: delta_pos
        gaussian: gauss_mean, gauss_std
        off: -
        """
        # Define wavelengths
        self.wavelengths = wavelengths

        # Thickness of CIGS layer and integration step size
        self.d = d
        self.dz = dz
        self.z_num = round(self.d/self.dz)+1
        self.z = np.linspace(0, self.d, self.z_num).reshape((self.z_num, 1))

        # Excitation laser properties
        self.wave_laser = wave_laser

        # Define refractive indices evaluated at the defined wavelengths
        if opt_data is None:
            nk_air = pd.read_csv(path.join(path_to_file, 'files', "nk_air.csv"))
            nk_CIGS = pd.read_csv(path.join(path_to_file, 'files', "nk_CIGS.csv"))
            nk_Mo = pd.read_csv(path.join(path_to_file, 'files', "nk_Mo.csv"))
            opt_data = {'air': nk_air, 'CIGS': nk_CIGS, 'Mo': nk_Mo}
        self.opt_data = {}
        for key in opt_data:
            if type(opt_data[key]) is not list:
                scaling = 1e-9
                n = interp1d(opt_data[key].iloc[:, 0]*scaling,
                             opt_data[key].iloc[:, 1], kind='linear')
                k = interp1d(opt_data[key].iloc[:, 0]*scaling,
                             opt_data[key].iloc[:, 2], kind='linear')
            else:
                scaling = self.scaling(opt_data[key][1])
                n = interp1d(opt_data[key][0].iloc[:, 0]*scaling,
                             opt_data[key][0].iloc[:, 1], kind='linear')
                k = interp1d(opt_data[key][0].iloc[:, 0]*scaling,
                             opt_data[key][0].iloc[:, 2], kind='linear')
            if key == 'CIGS':
                self.abs_CIGS_laser = (4*np.pi*k(self.wave_laser)
                                       / self.wave_laser)
            if key == 'Mo':
                self.opt_data[key] = [n(self.wavelengths),
                                      k_Mo*k(self.wavelengths)]
            else:
                self.opt_data[key] = [n(self.wavelengths), k(self.wavelengths)]

        # Include scalar scattering theory to account for roughness
        if rough is not None:
            self.roughness = rough
            self.rough_scaling = True
        else:
            self.rough_scaling = False

        # Angle of incident light
        self.theta_i = theta_i
        # Angle in CIGS
        self.theta_CIGS = np.arcsin(self.opt_data['air'][0]
                                    * np.sin(self.theta_i)
                                    / self.opt_data['CIGS'][0])

        # Define complex refractive indices
        self.N_air = self.opt_data['air'][0] + self.opt_data['air'][1]*1j
        self.N_CIGS = self.opt_data['CIGS'][0] + self.opt_data['CIGS'][1]*1j
        self.N_Mo = self.opt_data['Mo'][0] + self.opt_data['Mo'][1]*1j

        # Compute the fresnel coefficients for the stack
        self.fres21 = self.fresnel(self.N_CIGS, self.N_air, self.theta_CIGS,
                                   self.rough_scaling)
        self.fres23 = self.fresnel(self.N_CIGS, self.N_Mo, self.theta_CIGS)

        # Define the wanted function for excitation and luminescemse
        self.excitation = self.excitationExp
        self.luminescence = self.lumiDelta
        self.params = params
        self.updateModel(luminescence)

        self.emission_prob_val = self.emissionProb()
        self.luminescence_val = self.luminescence()
        self.excitation_val = self.excitation()

        self.IF = np.zeros(wavelengths.shape)
        self.solve()

    def updateFresnel(self):
        """Update the Fresnel coefficients, if needed."""
        self.theta_CIGS = np.arcsin(self.opt_data['air'][0]
                                    * np.sin(self.theta_i)
                                    / self.opt_data['CIGS'][0])
        self.fres21 = self.fresnel(self.N_CIGS, self.N_air, self.theta_CIGS,
                                   self.rough_scaling)
        self.fres23 = self.fresnel(self.N_CIGS, self.N_Mo, self.theta_CIGS)

    def updateModel(self, luminescence):
        """Update the PL model and corresponding functions."""
        if luminescence == 'uniform':
            self.excitation = self.excitationExp
            self.luminescence = self.lumiUniform
        elif luminescence == 'delta':
            self.excitation = self.excitationExp
            self.luminescence = self.lumiDelta
        elif luminescence == 'gaussian':
            self.excitation = self.excitationExp
            self.luminescence = self.lumiGaussian

    def solve(self):
        """Solve for the inteference function."""
        self.luminescence_val = self.luminescence()
        self.excitation_val = self.excitation()
        IFintegrand = self.IFintegrand()
        IF_sum_sp = (np.trapz(IFintegrand[0], self.z, axis=0)
                     + np.trapz(IFintegrand[1], self.z, axis=0))
        self.IF = (IF_sum_sp)/(2*np.trapz(self.IFnorm(), self.z, axis=0))

    def excitationExp(self):
        """Define an exponential decay for the excitation function."""
        return np.exp(-self.abs_CIGS_laser*self.z/np.cos(self.theta_CIGS))

    def lumiUniform(self):
        """Define a uniform function."""
        return np.ones(self.z.shape)

    def lumiDelta(self):
        """Define a delta function for the luminescence."""
        abs_diff = np.abs(self.z-self.params[0])
        idx_pulse = np.argmin(abs_diff)
        return signal.unit_impulse(self.z_num, idx_pulse).reshape(self.z.shape)

    def lumiGaussian(self):
        """Define a gaussian function for the luminescence."""
        return np.exp(-(self.z-self.params[0])**2/(2*self.params[1]**2))

    def IFintegrand(self):
        """Define the integrand for the IF."""
        return self.emission_prob_val*self.luminescence_val*self.excitation_val

    def IFnorm(self):
        """Define the normalisation integrand for the IF."""
        return self.luminescence_val*self.excitation_val

    def emissionProb(self):
        """Compute the emission probability."""
        delta = (4*np.pi*self.N_CIGS*np.cos(self.theta_CIGS)
                 * (self.d-self.z)/self.wavelengths)
        phi = (4*np.pi*self.N_CIGS*np.cos(self.theta_CIGS)
               * self.d/self.wavelengths)

        P_s = (self.opt_data['CIGS'][0]/(8*np.pi*self.opt_data['air'][0])
               * self.absSq(self.fres21[2])*self.absSq(1+self.fres23[0]
               * np.exp(delta*1j))/self.absSq(1-self.fres21[0]*self.fres23[0]
               * np.exp(phi*1j)))
        P_p = (self.opt_data['CIGS'][0]/(8*np.pi*self.opt_data['air'][0])
               * self.absSq(self.fres21[3])*(1+self.absSq(self.fres23[1])
               * np.exp(-2*delta.imag)+2*(self.fres23[1]*np.exp(delta*1j)).real
               * np.cos(2*self.theta_CIGS))
               / (self.absSq(1-self.fres21[1]*self.fres23[1]*np.exp(phi*1j))))
        return np.array([P_s, P_p])

    def fresnel(self, N1, N2, theta1, rough=False):
        """Compute the reflection and transmission coefficients at an interface
        between media 1 and 2."""
        theta2 = np.arcsin(N1.real*np.sin(theta1)/N2.real)
        cos_theta1 = np.cos(theta1)
        cos_theta2 = np.cos(theta2)
        r_s = (N1*cos_theta1-N2*cos_theta2)/(N1*cos_theta1+N2*cos_theta2)
        r_p = (N1*cos_theta2-N2*cos_theta1)/(N1*cos_theta2+N2*cos_theta1)
        t_s = (2*N1*cos_theta1)/(N1*cos_theta1+N2*cos_theta2)
        t_p = (2*N1*cos_theta2)/(N1*cos_theta2+N2*cos_theta1)
        if rough:
            r_scale = np.exp(-0.5*(4*np.pi*self.roughness*N1*cos_theta1
                                   / self.wavelengths)**2)
            t_scale = np.exp(-0.5*(2*np.pi*self.roughness
                                   * (N1*cos_theta1-N2*cos_theta2)
                                   / self.wavelengths)**2)
        else:
            r_scale = 1
            t_scale = 1
        return (r_scale*r_s, r_scale*r_p, t_scale*t_s, t_scale*t_p)

    def absSq(self, complex):
        """Compute the squared magnitude of a complex number."""
        return (complex*complex.conjugate()).real

    def scaling(self, prefix):
        """Returns the scaling corresponding to the input prefix.
        Default is nm."""
        if prefix == 'um':
            return 1e-6
        elif prefix == 'm':
            return 1
        else:
            return 1e-9
