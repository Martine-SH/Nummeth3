# -*- coding: utf-8 -*-
"""
Joost Gerlagh, 1433520
Numerical Methods
Part 3, Task M1
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
#%%
"Git should git gut fr fr"
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
        self.Kappa  = 10**-3 # Thermal diffusion coefficient (m^2/s)
        self.T0     = 200 # K, initial temperature rod
        self.T1     = 300 # K, temperature of rod at x = 0 during simulation 


# Throughout the code, I will try and do my best to use i when talking about
# the index of grid points and n when talking about the index of timesteps

#%% Theoretical solution

# First we define the theoretical solution
def T_Theory(xs, t, T0, T1, kappa):
    T = T0 + (T1 - T0) * special.erfc(xs / np.sqrt(4 * kappa * t))
    if xs[0] == 0 and t == 0: # at (0,0) there appears 0/0 in the exponent.
    # Luckily, we now what it's supposed to be
        T[0] = T1
    return T
# Note that, given the structure I want the output to have (namely a 2D array 
# Result of dimensions (nt, nx) where Result[n,i] = T_(n,i)), the function above
# takes an array of x-values at a (scalar) time t.

# This function takes arrays of the requested times (ts) and locations (xs)
# and then immediately calculates the temperature at all of them
# Here, Ts[n, i] will be T(t = t_n, x = x_i)
# (NB: n and i then start at 0 and end at nx - 1 and nt - 1 respectively)
def Theory(ts, xs, nt, nx, T0, T1, kappa):
    Ts = np.zeros((nt, nx))
    for n,t in enumerate(ts):
        Ts[n] = T_Theory(xs, t, T0, T1, kappa) 
    return Ts

#%% Defining a functions that calculates dT/dt

# Here we use our PDE which states dT/dt = κ * d^2T/dx^2
# We calculate the spatial derivative using finite differences:
# [d^2T/dx^2]_(n,i) = (T_(n,i+1) - 2T_(n,i) + T_(n, i - 1)) / (Δx)^2
def dT_dt_CD(Tn, nx, dx, kappa):
    d2Tdx2 = np.zeros(nx)
    d2Tdx2[1:-1] = (Tn[2:] - 2 * Tn[1:-1] + Tn[:-2]) / dx**2
    # 
    # We fix T(x = 0, t > 0) = T1, so that derivative just stays 0
    # Similar for T(x = L (-dx), t > 0) = T0
    #??? Or we just use T(x = L, t) = T_Theory(x = L, t)
    dTdt = kappa * d2Tdx2
    return dTdt

#TODO: write spectral derivative function
# def dT_dt_SP()

#%% Defining integrating functions 
# In the following functions, I used the following conventions:
    # Tn: an array of the current temperatures, i.e. at timestep n
    # T_prev: "         " previous "                           " n - 1
    # T_new: "          " next "                               " n + 1
# Note that all the functions take both T_prev and Tn as input, even though most
# use just the one. This is such that all functions have the same input, which 
# will be make it easier to write out a more general simulation function later.
# Also note that while Theory immediately yields the full soltion, these 
# functions all just yield intermediate solutions

# The following two functions need T_prev. 
# So they takes as input Ts, an array of the form [T_prev, Tn], and return a 
# similar array for the next timestep.
def Adams_Bashforth(Ts, nx, dx, dt, kappa, dT_dt):
    [T_prev, Tn] = Ts
    T_new = Tn + dt / 2 * (3 * dT_dt(Tn, nx, dx, kappa) - dT_dt(T_prev, nx, dx, kappa))
    return [Tn, T_new]

def Leap_frog(Ts, nx, dx, dt, kappa, dT_dt):
    [T_prev, Tn] = Ts
    T_new = T_prev + 2 * dt * dT_dt(Tn, nx, dx, kappa)
    return [Tn, T_new]

# The following three functions just need Tn, and "simply" return T_new (= T_(n+1))

def Crank_Nicholson(Tn, nx, dx, dt, kappa, dT_dt):
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
    
def Euler_forward(Tn, nx, dx, dt, kappa, dT_dt):
    T_new = Tn + dt * dT_dt(Tn, nx, dx, kappa)
    return T_new

def Runge_Kutta4(Tn, nx, dx, dt, kappa, dT_dt):
    # Note that dT/dt = κ d^2T/dx^2, which does not explicitly depend on time,
    # which is why we don't see any of the different arguments of time in the k's
    k1 = dT_dt(Tn, nx, dx, kappa)
    k2 = dT_dt(Tn + dt / 2 * k1, nx, dx, kappa)
    k3 = dT_dt(Tn + dt / 2 * k2, nx, dx, kappa)
    k4 = dT_dt(Tn + dt * k3, nx, dx, kappa)
    T_new = Tn + dt / 6 * (k1 + 2 * (k2 + k3) + k4)
    return T_new
    

#%% Defining a simulation functions
# Simulations can be run be calling the following function
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
    
    PhysC = PhysConstants() # load physical constants in `self`-defined variable PhysC
    
    # (start code)
    if DiffMethod == 'CD':
        dT_dt = dT_dt_CD
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
    

    Result_theory = Theory(Time, Xaxis, nt, nx, PhysC.T0, PhysC.T1, PhysC.Kappa)
    # The theoretical solution can just be calculated directly 
    # for every pair (x, t)
    
    Var = np.zeros(nt)
    
    if TimeSteppingMethod == "Theory":
        return Time, Xaxis, Result_theory, Var
        
    else:
        Result = np.zeros((nt, nx)) 
        # This will be the array of temperatures, where in the end
        # Results[n,i] will give T at time t_n (= n*dt) and location x_i (= i*dx)        
        Tn = np.full(nx, PhysC.T0) 
        Tn[0] = PhysC.T1
        # These are the temperatures at t = 0, and will during the simulation
        # be the temperatures t = t_n     
        Result[0] = Tn.copy()

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
            T_prev = Tn.copy()
            Tn = Euler_forward(Tn, nx, dx, dt, PhysC.Kappa, dT_dt)
            Result[1] = Tn.copy()
            Var[1] = np.var(Tn - Result_theory[1], mean = 0).copy()
            
            for n in range(2, nt):
                [T_prev, Tn] = integrator([T_prev, Tn], nx, dx, dt,
                                          PhysC.Kappa, dT_dt)
                Result[n] = Tn.copy()
                Var[n] = np.var(Tn - Result_theory[n], mean = 0).copy()
        else:        
            if TimeSteppingMethod == "EF":
                integrator = Euler_forward
            elif TimeSteppingMethod == "CN":
                integrator = Crank_Nicholson
            elif TimeSteppingMethod == "RK4":
                integrator = Runge_Kutta4
                
            for n in range(1, nt):
                Tn = integrator(Tn, nx, dx, dt, PhysC.Kappa, dT_dt)
                Result[n] = Tn.copy()
                Var[n] = np.var(Tn - Result_theory[n], mean = 0).copy()
                
        return Time, Xaxis, Result, Var

#%% Setting paramaters for simulation
#  TODO Let's for now consider a ... m rod split into ... equal parts, such that:
L = 1 # m
nx = 10**2
dx = L / nx # m
# TODO Now we use the fact that we fix the ratio(s) κ * Δt / Δx^2
ratios = np.array([0.25])
# ratios = np.linspace(0.25, 0.5, num = 6)
# From that we calculate the Δt's
dts = ratios * dx**2 / PhysConstants().Kappa
# TODO Now let's say we want to consider a ... amount timesteps also
nt = 10**3
TotalTime = nt * dts

#%% Running a simulation
#TODO Later this also gotta be for 2 Δt's and also for different derivatives ig
All_Results = {} # We create a dictionary to add all our simulation results to
# T_th = np.array([])
for j, dt in enumerate(dts):
    for DiffMethod in ["CD"]: # TODO add SP
        for TimeSteppingMethod in ["Theory", "EF", "AB", "RK4", "CN"]: #, "LF", "CN"]:
            Time, Xaxis, Result, Var = Task1_caller(L, nx, TotalTime[j], dt,
                                               TimeSteppingMethod, DiffMethod)
            All_Results[(TimeSteppingMethod, dt, DiffMethod)] = {
                                             "Time": Time,
                                             "Xaxis": Xaxis,
                                             "Temperatures": Result,
                                             "Variances": Var
                                             }
            # if TimeSteppingMethod == "Theory":
            #     T_th = [dt, Result]


#%% Plotting the solutions
# Plotting the entire solution would take a lot of time and is also hard to visualize.
# So I wrote a function that makes Ng graphs, with constant time intervals between
# them (and including t = 0 and t = TotalTime)
def T_x_plot(Results, Ng, nt, TSM = False, dt = False, DM = False):
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
                Temperatures = Result["Temperatures"]
                tg = Time[int(ng)]
                Tgs = Temperatures[int(ng)]
                plt.plot(Xaxis, Tgs, '.', label = f'TSM = {TimeSteppingMethod},'
                                                  f'DM = {DiffMethod}, '
                                                  f'$Δt$ = {dtg:.2e} s')
                plt.title(f'$T(t = {tg:.2e}, x)$')
            
            elif TSM != False and TimeSteppingMethod == TSM:
                Time = Result["Time"]
                Xaxis = Result["Xaxis"]
                Temperatures = Result["Temperatures"]
                tg = Time[int(ng)]
                Tgs = Temperatures[int(ng)]
                plt.plot(Xaxis, Tgs, '.', label = f'DM = {DiffMethod}, '
                                                  f'$Δt$ = {dtg:.2e} s')
                plt.title(f'$T(t = {tg:.2e}, x)$, TSM = {TSM}')
            
            elif dt != False and dtg == dt:
                Time = Result["Time"]
                Xaxis = Result["Xaxis"]
                Temperatures = Result["Temperatures"]
                tg = Time[int(ng)]
                Tgs = Temperatures[int(ng)]
                plt.plot(Xaxis, Tgs, '.', label = f'Int = {TimeSteppingMethod},'
                                                  f'DM = {DiffMethod}')
                plt.title(f'$T(t = {tg:.2e}, x), Δt = {dt:.2e}$ s')
            
            elif DM != False and DiffMethod == DM:
                Time = Result["Time"]
                Xaxis = Result["Xaxis"]
                Temperatures = Result["Temperatures"]
                tg = Time[int(ng)]
                Tgs = Temperatures[int(ng)]
                plt.plot(Xaxis, Tgs, '.', label = f'Int = {TimeSteppingMethod},'
                                                  f'$Δt$ = {dtg:.2e} s')
                plt.title(f'$T(t = {tg:.2e}, x)$, DM = {DM}')
        
        plt.xlim(0, L)
        # plt.ylim(PhysConstants().T0, PhysConstants().T1)
        plt.grid()
        plt.xlabel('$x$ (m)')
        plt.ylabel('T (K)')
        plt.legend()
        plt.show()

# for dtg in dts:
#     T_x_plot(All_Results, 1, nt, dt = dtg)
#%%
def plot_Var(Results):
    plt.figure()
    for (TimeSteppingMethod, dt, DiffMethod), Result in Results.items():
        Time = Result["Time"]
        Variances = Result["Variances"]
        plt.plot(Time, Variances, label = f'TMS = {TimeSteppingMethod}, $Δt$ = {dt} s')
        plt.xlabel("$t$ (s)")
        plt.ylabel("$σ^2$")
        plt.legend()
        plt.grid()
        plt.xlim(0, Time[-1])
    plt.show()
        
plot_Var(All_Results)
#%% 
# Delta = np.array([])
# for (TimeSteppingMethod, dt, DiffMethod), Result in All_Results.items():
#     if TimeSteppingMethod != "Theory":
#         T_TMS = Result["Temperatures"]
#         for dt_th, T_theory in T_th:
#             if dt_th == dt:
#                 DeltaT = T_th[-1] - T_TMS
#                 sig = np.zeros(nt)
#                 var = np.zeros(nt)
#                 for n, DeltaTn in enumerate(DeltaT):
#                     var_n = np.var(DeltaTn)
#                     sig_n = np.sqrt(var_n)
#                     sig[n] = sig_n
#                     var[n] = var_n

#%%
"""
First look, with L = 1 m, nx = 10**2, ratio = 0.5 (dt = 5 * 10*-5 s), nt = 10**5,
and T0 = 200 K, T1 = 3 K and κ = 1 m^2/s (I think, but I forgor tbh)
shows that RK4, AB and EF fully overlap, while significantly deviating from the 
theoretical curve. LF does weird. T(x = 0) alternates between 200 and 300, while
the others are either just 200 constantly or suddenly jump to very weird values...
Even when that problem is fixed, LF still explodes...
Tried ratios = np.linspace(0.1, 0.3, num = 5), did not help...
ratios = np.linspace(0.01, 0.1, num = 10) didn't either

Now for L = 1 m, nx = 10**3, ratio = 0.5 (dt = 50 s), nt = 10**5,
and T0 = 200 K, T1 = 3 K and κ = 10^-8 (a little more realistic value)
we see that EF and AB nicely overlap with the analytical solution, whereas RK4
produces a somehwat similar, but different, curve. LF still explodes.

Crank-Nicholson doesn't work for shit either. Fuckin awesome
Was also a me issue (switched A and B). But still does not work for shit...

RK4 seems to be stable (and following Theory quite well) for L = 1 m, nx = 10**2,
nt = 10**3 and T0 = 200 K, T1 = 3 K, κ = 10**-3 for ratio < 0.7 (Δt < 0.07 s).
AB seems, at the same conditions, to be stable for ratio \geq 0.5 (Δt = 0.05)
It doesn't necessarily agree with the Theory, contrary to most other methods.
EF ratio \geq 0.5 as well, agreeing quite well with Theory in that range too.
For CN I need to make the ratio ridiculously small to yield somehwat physical values..
Thing is that a small ratio also means a small TotalTime, so I it's safe to say
CN just isn't stable.
Basically same for LF. It only becomes somewhat stable when you make Δt so small 
that it barely does anything.
Nevermind I made a mistaky, CN seems to work after all. In fact, it seems to be 
the most stable of all. It seems to be stable for ratios up to 5 (Δt = 0.5 s),
although at that point Theory and simulation become harder to compare:
for the simulations we fixed the end at x = x_nx to be T0, which we didn't do in
the Theoretical solution. So as the ratios increase, Δt increases, TotalTime increases,
and thus also T(x = x_nx, t = t_nt). After ratio \approx 3, the shape of the 
CN simulation at t = t_nt barely seems to change,in fact appearing almost linear.

Plotting everything for ratios 0.25 and 0.5 with the other parameters as above, 
shows that all methods seem to agree relatively well with Theory, except for AB.

There waws a mistake in AB, that is fixed. It's now stable until raio is 0.25
"""