# Common functions for LISA data analysis
# 02024-07-29 CE (JD 2460521)
# Peyton Johnson, Grant David Meadors

import numpy as np
from itertools import combinations
from functools import cached_property

# Constants (SI units)
c = 299792458.      # speed of light (m/s)
AU = 149597870700.  # Astronomical Unit (m)
year = 31558149.8   # year (s)


# Functions to generate uncorrelated noise channels
def tdi_A(X, Y, Z):
    return (Z - X) / np.sqrt(2.)

def tdi_E(X, Y, Z):
    return (X - 2.*Y + Z) / np.sqrt(6.)

def tdi_T(X, Y, Z):
    return (X + Y + Z) / np.sqrt(3.)

def tdi_AET(tdi):
    return np.vstack((tdi[0], tdi[1], tdi[2],
                      tdi_A(tdi[0], tdi[1], tdi[2]),
                      tdi_E(tdi[0], tdi[1], tdi[2]),
                      tdi_T(tdi[0], tdi[1], tdi[2])))


class Orbit:
    def __init__(self, t, L, lm=0., kp=0.):
        """
        A class for constructing LISA satellite orbits

        Parameters
        ==========
        t : float array
            times to evaluate the orbital positions at

        L : float
            arm length [m]

        lm : float
            lambda, initial pinwheel phase [rad]

        kp: float
            kappa, inital orbital phase [rad]
        """
        self.t = t 
        self.L = L
        self.lm = lm
        self.kp = kp

    @cached_property
    def a(self):
        return AU

    @cached_property
    def T(self):
        return year

    @cached_property
    def doppler_scale(self):
        return (2. * np.pi * self.a / self.T) / c

    @cached_property
    def alpha(self):
        return np.tile(2.*np.pi*self.t/self.T + self.kp, (3, 1))

    @cached_property
    def beta(self):
        return ((2.*np.pi/3.)*np.arange(3) + self.lm).reshape((3, 1))

    @cached_property
    def ecc_fact(self):
        return self.L / (4. * np.sqrt(3.))

    @cached_property
    def x(self):
        return self.ecc_fact * (np.cos(2.*self.alpha-self.beta) -
               3.*np.cos(self.beta)) + self.a*np.cos(self.alpha)

    @cached_property
    def y(self):
        return self.ecc_fact * (np.sin(2.*self.alpha-self.beta) -
               3.*np.sin(self.beta)) + self.a*np.sin(self.alpha)

    @cached_property
    def z(self):
        return -(self.L / 2.) * np.cos(self.alpha - self.beta)
        
    @cached_property
    def R(self):
        return np.stack([self.x, self.y, self.z], axis=1)

    @cached_property
    def n(self):
        n = dict()
        for r, s in combinations(np.arange(3), 2):
            n[f'{r+1}{s+1}'] = (self.R[r] - self.R[s]) / self.L
        return n
        
    @cached_property
    def nn(self):
        nn = dict()
        for (r, s), n_rs in self.n.items():
            nn[r+s] = np.einsum('i...,j...', n_rs, n_rs)
        return nn


class Binary:
    def __init__(self, A, incl, f0, fdot, phi0, psi, lat, lon):
        """
        A class containting parameters of a white dwarf binary system

        Parameters
        ==========
        A : float
            GW amplitude 

        incl : float
            inclination with respect to observer [rad]

        f0 : float
            initial GW frequency [Hz]

        fdot : float
            rate of change of GW frequency [Hz/s]

        phi0 : float
            initial phase [rad]

        psi : float
            polarization angle [rad]

        lat : float
            ecliptic latitude [rad]

        lon : float
            ecliptic longitude [rad] 

        Properties
        ==========
        k : float array
            GW propagation unit vector

        u : float array
            orthonormal basis vector to 'k' and 'v'

        v : float array
            orthonormal bassis vector to 'u' and 'v'
        
        eps_plus : 2-d float array
            plus polarization tensor in the source frame

        eps_cross : 2-d float array
            cross polarization tensor in the source frame

        e_plus : 2-d float array
            plus polarization tensor in the detector frame

        e_cross : 2-d float array
            cross polarization tensor in the detector frame

        A_plus : float
            plus polarization amplitude coefficient

        A_cross : float
           cross polarization amplitude coefficient
        """
        self.A = A
        self.incl = incl
        self.f0 = f0
        self.fdot = fdot
        self.phi0 = phi0
        self.psi = psi
        self.lat = lat
        self.lon = lon

    @cached_property
    def k(self):
        return -np.array([np.cos(self.lat)*np.cos(self.lon),
                          np.cos(self.lat)*np.sin(self.lon),
                          np.sin(self.lat)])

    @cached_property
    def u(self):
        return np.array([np.sin(self.lon), -np.cos(self.lon), 0.])

    @cached_property
    def v(self):
        return np.array([-np.sin(self.lat)*np.cos(self.lon),
                         -np.sin(self.lat)*np.sin(self.lon),
                          np.cos(self.lat)])

    @cached_property
    def eps_plus(self):
        return np.outer(self.u, self.u) - np.outer(self.v, self.v)

    @cached_property
    def eps_cross(self):
        return np.outer(self.u, self.v) + np.outer(self.v, self.u)

    @cached_property
    def e_plus(self):
        return np.cos(2.*self.psi)*self.eps_plus + \
               np.sin(2.*self.psi)*self.eps_cross

    @cached_property
    def e_cross(self):
        return -np.sin(2.*self.psi)*self.eps_plus + \
                np.cos(2.*self.psi)*self.eps_cross

    @cached_property
    def A_plus(self):
        return self.A*(1. + np.cos(self.incl)**2.)

    @cached_property
    def A_cross(self):
        return -2.*self.A*np.cos(self.incl)



class Waveform():
    def __init__(self, t, binary, orbit):
        """
        A class for generating the strain TDI signals generated by LISA
        in response to a galactic white dwarf binary system

        Parameters
        ==========
        t : float array
            array of times to compute the waveform values at

        binary : Binary object
            container for binary system parameters and derived values

        orbit : Orbit object
            container for orbital elements and derived quantities
        """
        self.t = t
        self.binary = binary
        self.orbit = orbit

    def phi(self, t):
        # Calculate the phase evolution of the binary
        return -self.binary.phi0 + 2. * np.pi * \
               (self.binary.f0*t + 0.5*self.binary.fdot*(t**2))

    @cached_property
    def kdR(self):
        # Compute the component of the position vectors along k
        return np.dot(self.binary.k, self.orbit.R)

    @cached_property
    def omL(self):
        # Compute the ratio of signal frequency to transfer frequency
        return 2. * np.pi * self.orbit.L / c * \
               (self.binary.f0 + self.binary.fdot*self.t)

    @cached_property
    def kdn(self):
        # Compute the component of the link unit vectors along k
        kdn = dict()
        for r, s in combinations(['1', '2', '3'], 2):
            kdn[r+s] = np.dot(self.binary.k, self.orbit.n[r+s])
        return kdn

    @cached_property
    def F_plus(self):
        F_plus = dict()
        for (r, s), nn in self.orbit.nn.items():
            F_plus[r+s] = np.einsum('...ij,...ij', nn,
                                    self.binary.e_plus)
        return F_plus

    @cached_property
    def F_cross(self):
        F_cross = dict()
        for (r, s), nn in self.orbit.nn.items():
            F_cross[r+s] = np.einsum('...ij,...ij', nn,
                                     self.binary.e_cross)
        return F_cross

    @cached_property
    def Ups(self):
        ups = dict()
        for (r, s), kdn_rs in self.kdn.items():
            arg = 0.5*self.omL*np.array([1.-kdn_rs, 1.+kdn_rs])
            amp = np.sin(arg) / arg
            exp_arg = np.exp(-1.j*arg)
            exp_omL = np.exp(-1.j*self.omL)
            ups[r+s] = (amp[0] + amp[1]*exp_omL) * exp_arg[0]
            ups[s+r] = (amp[1] + amp[0]*exp_omL) * exp_arg[1]
        return ups

    def construct_tdi(self, node, link_1, link_2):
        slink_1 = ''.join(sorted(list(link_1)))
        slink_2 = ''.join(sorted(list(link_2)))
        amp = self.binary.A_plus * \
              (self.F_plus[slink_1]*self.Ups[link_1] - \
               self.F_plus[slink_2]*self.Ups[link_2]) + \
              1.j*self.binary.A_cross * \
              (self.F_cross[slink_1]*self.Ups[link_1] - \
               self.F_cross[slink_2]*self.Ups[link_2])
        p_arg = self.phi(self.t - self.kdR[node-1]/c) - self.omL
        return self.omL*np.sin(self.omL) * amp * np.exp(1.j*p_arg)

    @cached_property
    def X_tdi(self):
        return self.construct_tdi(1, '13', '12')

    @cached_property
    def Y_tdi(self):
        return self.construct_tdi(2, '21', '23')

    @cached_property
    def Z_tdi(self):
        return self.construct_tdi(3, '32', '31')

    @cached_property
    def A_tdi(self):
        return tdi_A(self.X_tdi, self.Y_tdi, self.Z_tdi)

    @cached_property
    def E_tdi(self):
        return tdi_E(self.X_tdi, self.Y_tdi, self.Z_tdi)

    @cached_property
    def T_tdi(self):
        return tdi_T(self.X_tdi, self.Y_tdi, self.Z_tdi)


    @cached_property
    def tdi_2_prefactor(self):
        return 2.j * np.sin(2*self.omL) * np.exp(-2.j*self.omL)
    @cached_property
    def X_tdi_2(self):
        return self.tdi_2_prefactor * self.X_tdi

    @cached_property
    def Y_tdi_2(self):
        return self.tdi_2_prefactor * self.Y_tdi

    @cached_property
    def Z_tdi_2(self):
        return self.tdi_2_prefactor * self.Z_tdi

    @cached_property
    def A_tdi_2(self):
        return self.tdi_2_prefactor * self.A_tdi

    @cached_property
    def E_tdi_2(self):
        return self.tdi_2_prefactor * self.E_tdi

    @cached_property
    def T_tdi_2(self):
        return self.tdi_2_prefactor * self.T_tdi


    @cached_property
    def a_ant(self):
        a_ant = dict()
        for (r, s), nn in self.orbit.nn.items():
            a_ant[r+s] = np.einsum('...ij,...ij', nn,
                                   self.binary.eps_plus)
        return a_ant

    @cached_property
    def b_ant(self):
        b_ant = dict()
        for (r, s), nn in self.orbit.nn.items():
            b_ant[r+s] = np.einsum('...ij,...ij', nn,
                                   self.binary.eps_cross)
        return b_ant

    @cached_property
    def a_ts(self):
        a_tdi = np.zeros((6, self.t.shape[0]), dtype=np.double)
        a_tdi[0] = self.a_ant['13'] - self.a_ant['12'] # 13 - 12
        a_tdi[1] = self.a_ant['12'] - self.a_ant['23'] # 21 - 23
        a_tdi[2] = self.a_ant['23'] - self.a_ant['13'] # 32 - 31
        return tdi_AET(a_tdi)

    @cached_property
    def b_ts(self):
        b_tdi = np.zeros((6, self.t.shape[0]), dtype=np.double)
        b_tdi[0] = self.b_ant['13'] - self.b_ant['12'] # 13 - 12
        b_tdi[1] = self.b_ant['12'] - self.b_ant['23'] # 21 - 23
        b_tdi[2] = self.b_ant['23'] - self.b_ant['13'] # 32 - 31
        return tdi_AET(b_tdi)

    @cached_property
    def ab_ts(self):
        return np.array([self.a_ts, self.b_ts])

    def resamp_t(self, sign_orbit=1., sign_binary=-1.):
        """ 
        Calculate the exact timeshift needed to detect a binary
        system with the given parameters, at specific times
        The sign_binary option is needed to look at spin-up
        The sign_orbit option controls the orbital delay
        """
        return sign_orbit * self.kdR / c + \
               sign_binary * np.pi * self.binary.fdot * self.t**2.
