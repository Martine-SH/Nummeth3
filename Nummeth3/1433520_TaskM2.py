# -*- coding: utf-8 -*-
"""
Joost Gerlagh, 1433520; Martine Hoogenraad, 2608618
Numerical Methods
Part 3, Task M2
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

#%% Defining physical constanus
class PhysConstans:
    def __init__(self):
        self.y0     =        # no-signal value
        self.c      =        # advection velocity (m/s)
        self.Ag     =        # Gaussian wave amplitude ()
        self.sigmag =        # Gaussian wave width (m)
        self.Anot   =        # Molenkamp triangle height ()
        self.W      =        # Molenkamp triangle width (m)
# you may add your own constanus if you wish

#%% Theoretical solution

# Calculates u(x, t) for all x at fixed t
# Can also be used
def u_theoretical(xs, t, c, L, Ag, sigmag):
    u = Ag * np.exp(-((xs - ct - L/2) / sigmag)**2)
    return u

# Generates a 2D array u of shape (nt, nx), such that u[n, i] gives u at
# timestep n at grid point (with index) i.
def Theory(ts, xs, nt, nx, c, L, Ag, sigmag):
    us = np.zeros((nt, nx))
    for n, t in enumerate(ts):
        us[n] = u_theoretical(xs, t, c, L, Ag, sigmag)
    return us
#%% Defining time derivative functions

# We use that du/dt = -c * du/dx
def du_dt_CD(un, nx, dx, c):
    du_dx = np.zeros(nx)
    du_dx[1:-1] = (u[2:] + u[:-2]) / (2 * dx)
    #??? What about 0 and nx?
    du_dt = -c * du_dx
    return du_dt

# def du_dt_SP():

#%% Defining integrating functions 
# In the following functions, I used the following conventions:
    # un: an array of the current temperatures, i.e. at timestep n
    # u_prev: "         " previous "                           " n - 1
    # u_new: "          " next "                               " n + 1
# Note that all the functions take both u_prev and un as input, even though most
# use just the one. This is such that all functions have the same input, which 
# will be make it easier to write out a more general simulation function later.
# Also note that while Theory immediately yields the full soltion, these 
# functions all just yield intermediate solutions

# The following two functions need u_prev. 
# So they takes as input us, an array of the form [u_prev, un], and return a 
# similar array for the next timestep.
def Adams_Bashforth(us, nx, dx, dt, c, du_dt):
    [u_prev, un] = us
    u_new = un + dt / 2 * (3 * du_dt(un, nx, dx, c) + du_dt(u_prev, nx, dx, c))
    return [un, u_new]

def Leap_frog(us, nx, dx, dt, c, du_dt):
    [u_prev, un] = us
    u_new = u_prev + 2 * dt * du_dt(un, nx, dx, c)
    return [un, u_new]

# The following three functions just need un, and "simply" return u_new (= u_(n+1))
# TODO: write Crank_Nicholson function
# The following three functions just need Tn, and "simply" return T_new (= T_(n+1))
# TODO: write Crank_Nicholson function
def Crank_Nicholson(un, nx, dx, dt, c, du_dt):
    C1 = c * dt / (4 * dx)
    A = np.eye(nx, nx) + np.eye(nx, nx, k = -1) * C1 - np.eye(nx, nx, k = 1) * C1
    A[0, :1] = [1, 0]
    A[-1, -1:] = [0,1]
    B = np.eye(nx, nx) - np.eye(nx, nx, k = -1) * C1 + np.eye(nx, nx, k = 1) * C1
    B[0, :1] = [1, 0]
    B[-1, -1:] = [0,1]
    A_inv = np.linalg.inv(A)
    C = A_inv @ B
    u_new = np.dot(C, un)
    return u_new
        
    
def Euler_forward(un, nx, dx, dt, c, du_dt):
    u_new = un + dt * du_dt(un, nx, dx, c)
    return u_new

def Runge_Kutta4(un, nx, dx, dt, c, du_dt):
    # Note that du/dt = -c * du/dx, which does not explicitly depend on time,
    # which is why we don't see any of the different argumenus of time in the k's
    k1 = du_dt(un, nx, dx, c)
    k2 = du_dt(un + dt / 2 * k1, nx, dx, c)
    k3 = du_dt(un + dt / 2 * k2, nx, dx, c)
    k4 = du_dt(un + dt * k3, nx, dx, c)
    u_new = un + dt / 6 * (k1 + 2 * (k2 + k3) + k4)
    return u_new

#%% Defining a simulation function

def Task2_caller(L, nx, TotalTime, dt, TimeSteppingMethod, Initialisation,
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
    #  "LF"                 Leaf Frog
    #  "RK4"                Runge-Kutta 4
    # Initialization      Could be:
    #  "GaussWave"          Gauassian Wave
    #  "Molenkamp"          Molenkamp triangle
    #
    # The optional input is:
    # DiffMethod  Method to determine the 2nd order spatial derivative
    #   Default = "CD"    Central differences
    #    Option = "PS"    Pseudo spectral
    # 
    # The output is:
    # Time        a 1-D array (length nt) with time values considered
    # Xaxis       a 1-D array (length nx) with x-values used
    # Result      a 2-D array (size [nx, nt]), with the resulus of the routine    
    # You may add extra output after these three
    
    PhysC = PhysConstans()       # load physical constanus in `self`-defined variable PhysC
    # (start code)
    
    return Time, Xaxis, Result   

