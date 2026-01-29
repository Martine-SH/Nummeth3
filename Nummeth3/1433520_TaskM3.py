# -*- coding: utf-8 -*-
"""
Joost Gerlagh, 1433520; Martine Hoogenraad 2608618
Numerical Methods
Part 3, Task M3
"""
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

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
def inital_states(sigma, x):
    A = (2*sigma/np.pi)**0.25
    psi0 = A*np.exp(-2*sigma*x**2)
    return psi0
    
#%% Defining a functions that calculates dT/dt

# Here we use our PDE which states dT/dt = κ * d^2T/dx^2
# We calculate the spatial derivative using finite differences:
# [d^2T/dx^2]_(n,i) = (T_(n,i+1) - 2T_(n,i) + T_(n, i - 1)) / (Δx)^2
def dpsi_dt_CD(psi_n, nx, x, dx, h , m, omega):
    d2psidx2 = np.zeros(nx, dtype=complex)
    d2psidx2[1:-1] = (psi_n[2:] - 2 * psi_n[1:-1] + psi_n[:-2]) / dx**2 
    
    d2psidx2[0] = (psi_n[1] - 2*psi_n[0] + psi_n[-1]) / dx**2
    d2psidx2[-1] = (psi_n[0] - 2*psi_n[-1] + psi_n[-2]) / dx**2
    # We fix Psi(x = 0, t > 0) = psi1, so that derivative just stays 0
    # Similar for Psi(x = L (-dx), t > 0) = psi0
    dpsidt = np.zeros(nx, dtype=complex)
    dpsidt = 1j * h / (2 * m) * d2psidx2 - 1j * m * omega**2 * x**2 * psi_n / (2 * h)
    
    return dpsidt

#%% Defining integrating functions 
# In the following functions, I used the following conventions:
    # p_sin: an array of the current wavefunctions, i.e. at timestep n
    # Psi_prev: "         " previous "                           " n - 1
    # Psi_new: "          " next "                               " n + 1
# Note that all the functions take both Psi_prev and p_sin as input, even though most
# use just the one. This is such that all functions have the same input, which 
# will be make it easier to write out a more general simulation function later.
# Also note that while Theory immediately yields the full soltion, these 
# functions all just yield intermediate solutions

# The following two functions need T_prev. 
# So they takes as input Ts, an array of the form [T_prev, Tn], and return a 
# similar array for the next timestep.
def Adams_Bashforth(psi_s, nx, x, dx, dt, h, m, omega, dpsi_dt):
    [psi_prev, psi_n] = psi_s
    psi_new = psi_n + dt / 2 * (3 * dpsi_dt(psi_n, nx, x, dx, h , m, omega) - dpsi_dt(psi_prev, nx, x, dx, h , m, omega))
    return [psi_n, psi_new]

def Leap_frog(psi_s, nx, x, dx, dt, h, m, omega, dpsi_dt):
    [psi_prev, psi_n] = psi_s
    psi_new = psi_prev + 2 * dt * dpsi_dt(psi_n, nx, x, dx, h , m, omega)
    return [psi_n, psi_new]

# The following three functions just need Tn, and "simply" return T_new (= T_(n+1))

def Crank_Nicholson(psi_n, nx, x, dx, dt, h, m, omega, dpsi_dt):
    # Note: I accidentally switched the names of B and A compared to the LN
    CT = 1j*h / dt#part on the left side of the SE
    C1 = h**2 * dt / (4 * m * dx**2) / CT
    C2 = m * omega **2 * x**2 * dt * 0.5 / CT

    A = np.eye(nx, dtype=complex) * (1 -2 * C1 + C2) + np.eye(nx, k = -1) * -C1 + np.eye(nx, k = 1) * -C1
    A[0, 0], A[0,1] = 1, 0
    A[-1,-2], A[-1, -1] = 0, 1
    B = np.eye(nx, dtype=complex) * (1 -2 * C1 + C2) - np.eye(nx, k = -1) * C1 - np.eye(nx, k = 1) * C1
    B[0, 0], B[0,1] = 1, 0
    B[-1, -2], B[-1, -1] = 0, 1
    B_inv = np.linalg.inv(B)
    C = B_inv @ A
    # T_new = np.dot(C, Tn)
    T_new = C @ psi_n
    return T_new
    
def Euler_forward(psi_n, nx, x, dx, dt, h, m, omega, dpsi_dt):
    psi_new = psi_n + dt * dpsi_dt(psi_n, nx, x, dx, h , m, omega)
    return psi_new

def Runge_Kutta4(psi_n, nx, x, dx, dt, h, m, omega, dpsi_dt):
    # Note that dT/dt = κ d^2T/dx^2, which does not explicitly depend on time,
    # which is why we don't see any of the different arguments of time in the k's
    k1 = dpsi_dt(psi_n, nx, x, dx, h , m, omega)
    k2 = dpsi_dt(psi_n + dt / 2 * k1, nx, x, dx, h, m, omega)
    k3 = dpsi_dt(psi_n + dt / 2 * k2, nx, x, dx, h, m, omega)
    k4 = dpsi_dt(psi_n + dt * k3, nx, x, dx, h, m, omega)
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
    
    PhysC = PhysConstants()   # load physical constanus in `self`-defined variable PhysC
    
    # (start code)
    if DiffMethod == 'CD':
        dpsi_dt = dpsi_dt_CD
    # elif DiffMethod == 'PS':
    #     dT_dt = dT_dt_PS
    
    Time = np.arange(0, TotalTime, dt) 
    # array of evenly spaced times with requested difference dt
    Xaxis = np.linspace(0, L, nx, endpoint = False) 
    # array of evenly spaced loactions with requested length nx
    nt = len(Time) 
    # number of timesteps we consider (NB if we integrate until t_N = Ndt, 
    # nt = N + 1)


    dx = L/nx # (constant) difference between two consecutvie grid points
    # x based on dx*(i-N/2)
    x = dx * (np.arange(nx) - nx/2)
    # x which works like Task1
    #x = np.linspace(-L/2, L/2, nx) 
    # This will be the array of wavefunctions, where in the end
    # Results[n,i] will give Psi at time t_n (= n*dt) and location x_i (= i*dx) 
    Result = np.zeros((nt, nx), dtype=complex) 
    psi_n = inital_states(PhysC.sigma, x)   
    Result[0] = psi_n.copy()
    

    
    if TimeSteppingMethod ==  "LF" or TimeSteppingMethod == "AB":
        # T_prev = np.full(nx, PhysC.T0) # the physical situation we're considering
        # is a rod at T = T0 and then suddenly at T1, we heat one end to
        # T = T1. So whe can consider the full rod at T0 as the previous state
        #??? This does NOT work as because of the way LF is written, T(x = 0)
        # will now jump between T0 and T1. So we do the first step using EF
        
        if TimeSteppingMethod == "LF":
            integrator = Leap_frog
        elif TimeSteppingMethod == "AB":
            integrator = Adams_Bashforth
        # We do the first step using forward Euler. We already define T_prev
        # with Tn at t = 0, as we will need both T_(n-1) (= T_prev) and Tn
        # for LF and AB
        #adding an initial value for psi_n
        psi_prev = psi_n.copy()
        psi_n = Euler_forward(psi_n, nx, x, dx, dt, PhysC.h, PhysC.m, PhysC.omega, dpsi_dt)
        Result[1] = psi_n.copy()
        
        for n in range(2, nt):
            [psi_prev, psi_n] = integrator([psi_prev, psi_n], nx, x, dx, dt,
                                      PhysC.h, PhysC.m, PhysC.omega, dpsi_dt)
            Result[n] = psi_n.copy()

    else:        
        if TimeSteppingMethod == "EF":
            integrator = Euler_forward
        elif TimeSteppingMethod == "CN":
            integrator = Crank_Nicholson
        elif TimeSteppingMethod == "RK4":
            integrator = Runge_Kutta4
            
        for n in range(1, nt):
            psi_n = integrator(psi_n, nx, x, dx, dt, PhysC.h, PhysC.m, PhysC.omega, dpsi_dt)
            Result[n] = psi_n.copy()
         
    return Time, Xaxis, Result  

#%% Setting paramaters for simulation
#  TODO Let's for now consider a ... m rod split into ... equal parts, such that:
L = 1 # m
nx = 10**3
dx = L / nx # m
# TODO Now we use the fact that we fix the ratio(s) κ * Δt / Δx^2
ratios = np.array([0.05])
# ratios = np.linspace(0.25, 0.5, num = 6)
# From that we calculate the Δt's
dts = ratios * PhysConstants().m * dx**2 / PhysConstants().h
# TODO Now let's say we want to consider a ... amount timesteps also
nt = 10**3
TotalTime = nt * dts

#%% Running a simulation
#TODO Later this also gotta be for 2 Δt's and also for different derivatives ig
All_Results = {} # We create a dictionary to add all our simulation results to
# T_th = np.array([])
for j, dt in enumerate(dts):
    for DiffMethod in ["CD"]: # TODO add SP
        for TimeSteppingMethod in ["EF", "AB", "RK4", "CN"]: #, "LF", "CN"]:
            Time, Xaxis, Result = Task3_caller(L, nx, TotalTime[j], dt,
                                               TimeSteppingMethod, DiffMethod)
            All_Results[(TimeSteppingMethod, dt, DiffMethod)] = {
                                             "Time": Time,
                                             "Xaxis": Xaxis,
                                             "Wavefunctions": Result,
                                             }
            # if TimeSteppingMethod == "Theory":
            #     T_th = [dt, Result]


#%% Plotting the solutions
# Plotting the entire solution would take a lot of time and is also hard to visualize.
# So I wrote a function that makes Ng graphs, with constant time intervals between
# them (and including t = 0 and t = TotalTime)
def Psi_x_plot(Results, Ng, nt, TSM = False, dt = False, DM = False):
    # dt, nt and DiffMethod are the same variables as before
    # If no input is given for dt or DM, the plot is made for all different 
    # possibilities. If a sepcific value/function is given, only that one is displayed.
    # Currently, only one of these inputs can be given unfortunately. Working on it.
    # Ng is the number of graphs the function should make
    # Results should be a dictionary featuring the Results of the simulations
    # in principal this will always be All_Results
    ngs = np.rint(np.linspace(0, nt - 1, num = Ng + 1))
    ngs = ngs[1:]
    # ngs will be the list of indices of time for which we will plot T(t_(ng); x)
    # This will at least include ng = 0 and ng = Nt (i.e. the first and final steps).
    # Although we cut out the initial situation, as that is non-interesting.
    # The nt - 1 is there because the indices of an array with nt elements go from
    # 0 to nt - 1.
    for ng in ngs:
        plt.figure()
        for (TimeSteppingMethod, dtg, DiffMethod), Result in Results.items():
            if dt == DM and dt == TSM: # i.e. if all are False and no input is given
                Time = Result["Time"]
                Xaxis = Result["Xaxis"]
                Psi = Result["Wavefunctions"]
                tg = Time[int(ng)]
                Psi_gs = Psi[int(ng)]
                plt.plot(Xaxis, Psi_gs, '.', label = f'TSM = {TimeSteppingMethod},'
                                                  f'DM = {DiffMethod}, '
                                                  f'$Δt$ = {dtg:.2e} s')
                plt.title(f'$Psi(t = {tg:.2e}, x)$')
            
            elif TSM != False and TimeSteppingMethod == TSM:
                Time = Result["Time"]
                Xaxis = Result["Xaxis"]
                Psi = Result["Wavefunctions"]
                tg = Time[int(ng)]
                Psi_gs = Psi[int(ng)]
                plt.plot(Xaxis, Psi_gs, '.', label = f'DM = {DiffMethod}, '
                                                  f'$Δt$ = {dtg:.2e} s')
                plt.title(f'$Psi(t = {tg:.2e}, x)$, TSM = {TSM}')
            
            elif dt != False and dtg == dt:
                Time = Result["Time"]
                Xaxis = Result["Xaxis"]
                Psi = Result["Wavefunctions"]
                tg = Time[int(ng)]
                Psi_gs = Psi[int(ng)]
                plt.plot(Xaxis, Psi_gs, '.', label = f'Int = {TimeSteppingMethod},'
                                                  f'DM = {DiffMethod}')
                plt.title(f'$Psi(t = {tg:.2e}, x), Δt = {dt:.2e}$ s')
            
            elif DM != False and DiffMethod == DM:
                Time = Result["Time"]
                Xaxis = Result["Xaxis"]
                Psi = Result["Wavefunctions"]
                tg = Time[int(ng)]
                Psi_gs = Psi[int(ng)]
                plt.plot(Xaxis, Psi_gs, '.', label = f'Int = {TimeSteppingMethod},'
                                                  f'$Δt$ = {dtg:.2e} s')
                plt.title(f'$Psi(t = {tg:.2e}, x)$, DM = {DM}')
        
        plt.xlim(0, L)
        # plt.ylim(PhysConstants().T0, PhysConstants().T1)
        plt.grid()
        plt.xlabel('$x$ (m)')
        plt.ylabel('Wavefunction')
        plt.legend()
        plt.show()
for dtg in dts:
    Psi_x_plot(All_Results, 1, nt, dt = dtg)

