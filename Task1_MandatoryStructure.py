#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:16:29 2024

@author: You - thus add in your name here
"""

# 1) add here the setting(s) you run the model

# but define the physical constants in this constants_class:
# Again, this is required to unify scripts to ease correcting it
class PhysConstants:
    def __init__(self):
        self.Kappa  =        # Thermal diffusion coefficient (m2/s)
        self.T0     =        # Initial temperature rod (K)
        self.T1     =        # Temperature of rod at x=0 for t>0
# you may add your own constants if you wish


# 2) add here supporting code 




# 3) Use the following routine to start simulations
def Task1_caller(L, nx, TotalTime, dt, TimeSteppingMethod, 
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
    
    PhysC = PhysConstants()       # load physical constants in `self`-defined variable PhysC
    # (start code)
    
    return Time, Xaxis, Result    


# 4) Code here (or in another file) the commands that use Task1_caller

# 5) Code here (or in another file) the analysis of the results and the plotting
