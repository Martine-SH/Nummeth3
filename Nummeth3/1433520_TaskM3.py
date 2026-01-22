# -*- coding: utf-8 -*-
"""
Joost Gerlagh, 1433520; Martine Hoogenraad 2608618
Numerical Methods
Part 3, Task M3
"""
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
#%%
"Git should giterer gut fr fr"
# A = np.eye(3) * 5 + np.eye(3, k = 1) * 3
# B = np.eye(3) * 4 + np.eye(3, k = 1) * 2
# E = [5, 6, 7]
# F = A @ E
# C = A @ B
# D = B @ A
# A[0, 0], A[0,1] = 1, 0
# A[-1,-2], A[-1, -1] = 0, 1


#%% Defining necessary constants

class PhysConstants:
    def __init__(self):
        self.h      = 1.0545718 *10**-34   #reduced Planck constant J*s/rad 
        self.m      = 9.11 * 10**-31 #mass electron in kilogram (kg)
        self.sigma  = 10 **-18 # width of wavefunction (given value)
        self.omega  = 10 **14  # width of well (chosen value can be changed)


# Throughout the code, I will try and do my best to use i when talking about
# the index of grid points and n when talking about the index of timesteps


#%%
def inital_states(sigma, nx):
    A = (2*sigma/np.pi)**0.25
    psi0 = A*np.exp(-2*sigma*nx**2)
    
#%% Defining a functions that calculates dT/dt

# Here we use our PDE which states dT/dt = κ * d^2T/dx^2
# We calculate the spatial derivative using finite differences:
# [d^2T/dx^2]_(n,i) = (T_(n,i+1) - 2T_(n,i) + T_(n, i - 1)) / (Δx)^2
def dpsi_dt_CD(psin, nx, dx, h , m):
    d2psidx2 = np.zeros(nx)
    d2psidx2[1:-1] = (psin[2:] - 2 * psin[1:-1] + psin[:-2]) / dx**2 
    # 
    # We fix Psi(x = 0, t > 0) = psi1, so that derivative just stays 0
    # Similar for Psi(x = L (-dx), t > 0) = psi0
    dpsidt = 1j * h / (2 * m) * d2psidx2 - 1j * m * omega**2 * nx**2 * psin[1:-1] / (2 * h)
    return dpsidt

#%% Defining integrating functions 
# In the following functions, I used the following conventions:
    # Psin: an array of the current wavefunctions, i.e. at timestep n
    # Psi_prev: "         " previous "                           " n - 1
    # Psi_new: "          " next "                               " n + 1
# Note that all the functions take both Psi_prev and Psin as input, even though most
# use just the one. This is such that all functions have the same input, which 
# will be make it easier to write out a more general simulation function later.
# Also note that while Theory immediately yields the full soltion, these 
# functions all just yield intermediate solutions

# The following two functions need T_prev. 
# So they takes as input Ts, an array of the form [T_prev, Tn], and return a 
# similar array for the next timestep.
def Adams_Bashforth(psi_s, nx, dx, dt, h, m, dpsi_dt):
    [psi_prev, psin] = psi_s
    psi_new = psin + dt / 2 * (3 * dpsi_dt(psin, nx, dx, h, m) - dpsi_dt(psi_prev, nx, dx, h, m))
    return [psin, psi_new]

def Leap_frog(psi_s, nx, dx, dt, h, m, dpsi_dt):
    [psi_prev, psin] = psi_s
    psi_new = psi_prev + 2 * dt * dpsi_dt(psin, nx, dx, h, m)
    return [psin, psi_new]

# The following three functions just need Tn, and "simply" return T_new (= T_(n+1))

def Crank_Nicholson(psi_n, nx, dx, dt, h, m, dpsi_dt):
    # Note: I accidentally switched the names of A and B compared to the LN
    C2 = kappa * dt / (2 * dx**2)
    A = np.eye(nx) * (1 - 2 * C2) + np.eye(nx, k = -1) * C2 + np.eye(nx, k = 1) * C2
    A[0, 0], A[0,1] = 1, 0
    A[-1,-2], A[-1, -1] = 0, 1
    B = np.eye(nx) * (1 + 2 * C2) - np.eye(nx, k = -1) * C2 - np.eye(nx, k = 1) * C2
    B[0, 0], B[0,1] = 1, 0
    B[-1, -2], B[-1, -1] = 0, 1
    B_inv = np.linalg.inv(B)
    C = B_inv @ A
    # T_new = np.dot(C, Tn)
    T_new = C @ Tn
    return T_new
    
def Euler_forward(psi_n, nx, dx, dt, h, m, dpsi_dt):
    psi_new = psi_n + dt * dpsi_dt(psi_n, nx, dx, h, m)
    return psi_new

def Runge_Kutta4(psi_n, nx, dx, dt, h, m, dpsi_dt):
    # Note that dT/dt = κ d^2T/dx^2, which does not explicitly depend on time,
    # which is why we don't see any of the different arguments of time in the k's
    k1 = dpsi_dt(psi_n, nx, dx, h, m)
    k2 = dpsi_dt(psi_n + dt / 2 * k1, nx, dx, h, m)
    k3 = dpsi_dt(psi_n + dt / 2 * k2, nx, dx, h, m)
    k4 = dpsi_dt(psi_n + dt * k3, nx, dx, h, m)
    psi_new = psi_n + dt / 6 * (k1 + 2 * (k2 + k3) + k4)
    return psi_new
    

#%% Defining a simulation functions
# Simulations can be run be calling the following function
def Task3_caller(L, nx, TotalTime, dt, TimeSteppingMethod, 
                 DiffMethod="CD"):
    # The mandatory input is:
    # L                   Length of domain to be modelled (m)
    # nx                  Number of gridpoint in the model domain
    # TotalTime           Total length of the simulation (s)
    # dt                  Length of each time step (s)
    # TimeSteppingMethod  Could be:
    #  "Theory"             Theoretical solution
    #  "AB"                 Adams-Bashforth
    #  "CN"                 Crank-Nicholson
    #  "EF"                 Euler Forward
    #  "LF"                 Leap Frog
    #  "RK4"                Runge-Kutta 4
    #
    # The optional input is:
    # DiffMethod  Method to determine the 2nd order spatial derivative
    #   Default = "CD"    Central differences
    #    Option = "PS"    Pseudo spectral
    # 
    # The output is:
    # Time        a 1-D array (length nt) with time values considered
    # Xaxis       a 1-D array (length nx) with x-values used
    # Result      a 2-D array (size [nx, nt]), with the results of the routine    
    # You may add extra output after these three
    
    PhysC = PhysConstans()       # load physical constanus in `self`-defined variable PhysC
    # (start code)
    
    return Time, Xaxis, Result   
