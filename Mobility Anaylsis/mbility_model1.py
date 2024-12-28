import numpy as np
from scipy.integrate import quad

class MobilityModel:
    """
    A library for calculating mobility in semiconductors under various scattering mechanisms.
    Includes methods for impurity scattering, optical phonon scattering, IRF mobility, etc.
    """

    # Physical constants
    q = 1.60217662e-19      # Elementary charge (C)
    m_e = 9.10938356e-31    # Electron mass (kg)
    eps0 = 8.85418782e-12   # Vacuum permittivity (F/m)
    hbar = 1.0545718e-34    # Reduced Planck constant (J.s)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)

    @staticmethod
    def a_func(x):
        """Lattice constant a as a function of composition x."""
        return (3.189 + (3.112 - 3.189) * x) * 1e-10  # in meters

    @staticmethod
    def c_func(x):
        """Lattice constant c as a function of composition x."""
        return (5.185 + (4.982 - 5.185) * x) * 1e-10  # in meters

    @staticmethod
    def m_star_func(x):
        """Effective mass as a function of composition x."""
        return (0.2 + (0.4 - 0.2) * x) * MobilityModel.m_e

    @staticmethod
    def epsilon_s_func(x):
        """Static dielectric constant as a function of composition x."""
        return 8.9 + (8.5 - 8.9) * x

    @staticmethod
    def epsilon_inf_func(x):
        """High-frequency dielectric constant as a function of composition x."""
        return 5.35 + (4.6 - 5.35) * x

    @staticmethod
    def volume_unit_cell(x):
        """Calculate the volume of a unit cell based on composition x."""
        a = MobilityModel.a_func(x)
        c = MobilityModel.c_func(x)
        return (np.sqrt(3) / 2) * a**2 * c

    @staticmethod
    def mobility_irf(T, n_2DEG, L, delta, m_star, epsilon_s):
        """
        Calculate IRF (interface roughness) mobility.

        Parameters:
            T (float): Temperature in Kelvin.
            n_2DEG (float): 2D electron density in m⁻².
            L (float): Thickness in meters.
            delta (float): Roughness parameter (dimensionless).
            m_star (float): Effective mass in kg.
            epsilon_s (float): Static dielectric constant.

        Returns:
            float: IRF mobility in cm²/V·s.
        """
        k_F = np.sqrt(2 * np.pi * n_2DEG)  # Fermi wavevector (1/m)
        q_TF = (m_star * MobilityModel.q**2) / (
            2 * np.pi * epsilon_s * MobilityModel.eps0 * MobilityModel.hbar**2
        )  # Thomas-Fermi wavevector (1/m)

        def integrand(u):
            numerator = u**4 * np.exp(-(L * k_F * u)**2)
            denominator = (u + q_TF / (2 * k_F))**2 * np.sqrt(1 - u**2)
            return numerator / denominator

        integral, _ = quad(integrand, 0, 1)

        tau_inv = (
            m_star * (MobilityModel.q**2 * delta * L * n_2DEG)**2 /
            (8 * (MobilityModel.eps0 * epsilon_s)**2 * MobilityModel.hbar**3)
        ) * integral

        mu_irf = MobilityModel.q / (tau_inv * m_star)
        return mu_irf * 1e4  # Convert to cm²/V·s

    @staticmethod
    def mobility_optical_phonon(T, n_2DEG, m_star, E_POP, epsilon_s, epsilon_inf):
        """
        Calculate optical phonon scattering mobility.

        Parameters:
            T (float): Temperature in Kelvin.
            n_2DEG (float): 2D electron density in m⁻².
            m_star (float): Effective mass in kg.
            E_POP (float): Optical phonon energy in eV.
            epsilon_s (float): Static dielectric constant.
            epsilon_inf (float): High-frequency dielectric constant.

        Returns:
            float: Optical phonon mobility in cm²/V·s.
        """
        omega_0 = E_POP * MobilityModel.q / MobilityModel.hbar
        Q_0 = np.sqrt(2 * m_star * omega_0 / MobilityModel.hbar)
        epsilon_p = 2 / (1 / epsilon_inf - 1 / epsilon_s)

        z = (
            np.pi * MobilityModel.hbar**2 * n_2DEG /
            (m_star * MobilityModel.k_B * T)
        )

        N_B = 1 / (np.exp(MobilityModel.hbar * omega_0 / (MobilityModel.k_B * T)) - 1)
        b = (
            8 * (epsilon_s / epsilon_p) *
            (MobilityModel.q**2 * n_2DEG) /
            (MobilityModel.hbar**2 * Q_0)
        )

        G_Q0 = (b * (8 * b**2 + 9 * Q_0**2) + 3 * Q_0**4) / (8 * (Q_0 + b)**3)
        F_z = 1 + (1 - np.exp(-z)) / z

        numerator = 2 * Q_0 * MobilityModel.hbar**2 * MobilityModel.eps0 * epsilon_p * F_z
        denominator = MobilityModel.q * m_star**2 * omega_0 * N_B * G_Q0

        mu_POP = numerator / denominator
        return mu_POP * 1e4  # Convert to cm²/V·s

    @staticmethod
    def mobility_imp(T, n_2DEG, N_imp, epsilon_s, m_star):
        """
        Calculate impurity scattering mobility.

        Parameters:
            T (float): Temperature in Kelvin.
            n_2DEG (float): 2D electron density in m⁻².
            N_imp (float): Impurity density in m⁻².
            epsilon_s (float): Static dielectric constant.
            m_star (float): Effective mass in kg.

        Returns:
            float: Impurity scattering mobility in cm²/V·s.
        """
        prefactor = (
            4 * (2 * np.pi)**(5 / 2) * MobilityModel.hbar**3 * (MobilityModel.eps0 * epsilon_s)**2
        ) / (m_star**2 * MobilityModel.q**3)

        mu_imp = prefactor * (n_2DEG**(3 / 2) / N_imp)
        return mu_imp * 1e4  # Convert to cm²/V·s
