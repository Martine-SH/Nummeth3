# -*- coding: utf-8 -*-
"""
Joost Gerlagh, 1433520
Numerical Methods
Part 3, Task M1
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Defining physical constants
class PhysConstants:
    def __init__(self):
        self.y0     = 1       # no-signal value #??? Why is this here?
        self.c      = 1       # advection velocity (m/s)
        self.Ag     = 0.5       # Gaussian wave amplitude ()
        self.sigmag = 0.2        # Gaussian wave width (m)
        self.Anot   = 0.5       # Molenkamp triangle height ()
        self.W      = 0.6       # Molenkamp triangle width (m)
# you may add your own constants if you wish

#%% Theoretical solutions/Initialisations

# Calculates u(x, t) for all x at fixed t (i.e. xs is an array, t a scalar)
# Can also be used for Initialisations
def u_Gauss(xs, t, L, c, Ag, sigmag):
    u = Ag * np.exp(-((xs - c * t - L/2) / sigmag)**2)
    return u

def u_Molenkamp(xs, t, nx, dx, L, A0, W):
     n1 = int(0.5 * W / dx) 
     n2 = int(W / dx)
     # as xs[n] = n * dx, this means that x[n1] < 0.5 * W < xs[n1 + 1]
     # and [n2] < W < xs[n2 + 1]
     u = np.zeros(nx)
     u[:n1 + 1] = 2 * A0 * xs[:n1 + 1] / W
     u[n1 + 1 : n2 + 1] = 2 * A0 * (1 - xs[n1 + 1 : n2 + 1] / W)
     return u


# Generates a 2D array u of shape (nt, nx), such that u[n, i] gives u at
# timestep n at grid point (with index) i.
def Theory_Gauss(ts, xs, nt, nx, L, c, Ag, sigmag):
    us = np.zeros((nt, nx))
    for n, t in enumerate(ts):
        us[n] = u_Gauss(xs, t, c, L, Ag, sigmag)
    return us
#%% Defining time derivative functions

# We use that du/dt = -c * du/dx
def du_dt_CD(un, nx, dx, c):
    du_dx = np.zeros(nx)
    du_dx[1:-1] = (un[2:] + un[:-2]) / (2 * dx)
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
    A = np.eye(nx) - np.eye(nx, k = -1) * C1 + np.eye(nx, k = 1) * C1
    A[0, 0], A[0,1] = 1, 0
    A[-1,-2], A[-1, -1] = 0, 1
    B = np.eye(nx) + np.eye(nx, k = -1) * C1 - np.eye(nx, k = 1) * C1
    B[0, 0], B[0,1] = 1, 0
    B[-1, -2], B[-1, -1] = 0, 1
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
    # DiffMethod  Method to determine the 1st order spatial derivative
    #   Default = "CD"    Central differences
    #    Option = "PS"    Pseudo spectral
    # 
    # The output is:
    # Time        a 1-D array (length nt) with time values considered
    # Xaxis       a 1-D array (length nx) with x-values used
    # Result      a 2-D array (size [nx, nt]), with the resulus of the routine    
    # You may add extra output after these three
    PhysC = PhysConstants() # load physical constants in `self`-defined variable PhysC
    
    # (start code)
    if DiffMethod == 'CD':
        du_dt = du_dt_CD
    # elif DiffMethod == 'PS':
    #     du_dt = du_dt_PS
    
    Time = np.arange(0, TotalTime, dt) 
    # array of evenly spaced times with requested difference dt
    Xaxis = np.linspace(0, L, nx, endpoint = False) 
    # array of evenly spaced loactions with requested length nx
    nt = len(Time) 
    # number of timesteps we consider (NB if we integrate until t_N = Ndt, 
    # nt = N + 1)
    dx = L/nx # (constant) difference between two consecutvie grid points
    
    if Initialisation == "GaussWave":
        Result_theory = Theory_Gauss(Time, Xaxis, nt, nx, L, PhysC.c, PhysC.Ag, PhysC.sigmag)
        un = u_Gauss(Xaxis, 0, L, PhysC.c, PhysC.Ag, PhysC.sigmag)
    elif Initialisation == "Molenkamp":
        Result_theory = Theory_Molenkamp()
        un = u_Molenkamp(Xaxis, 0, nx, dx, L, PhysC.A0, PhysC.W)
    # The theoretical solution can just be calculated directly for every pair (x, t)
    # un is the wave at t = 0. Throughout the simulation, un will be u(t = t_n).
    
    Var = np.zeros(nt)
    # In this array we will drop the variances/std's whatver we end up deciding on
    
    if TimeSteppingMethod == "Theory":
        return Time, Xaxis, Result_theory, Var
        
    else:    
        Result = np.zeros((nt, nx)) 
        # This will be the array of our values for u, where in the end
        # Results[n,i] will give u at time t_n (= n*dt) and location x_i (= i*dx)        
        
        Result[0] = un.copy()

        if TimeSteppingMethod ==  "LF" or TimeSteppingMethod == "AB":
            
            if TimeSteppingMethod == "LF":
                integrator = Leap_frog
            elif TimeSteppingMethod == "AB":
                integrator = Adams_Bashforth
            # We do the first step using forward Euler. We already define u_prev
            # with un at t = 0, as we will need both u_(n-1) (= u_prev) and un
            # for LF and AB
            u_prev = un.copy()
            un = Euler_forward(un, nx, dx, dt, PhysC.c, du_dt)
            Result[1] = un
            Var[1] = np.var(un - Result_theory[1], mean = 0).copy()
            
            for n in range(2, nt):
                [u_prev, un] = integrator([u_prev, un], nx, dx, dt, PhysC.c, du_dt)
                Result[n] = un.copy()
                Var[n] = np.var(un - Result_theory[n], mean = 0).copy()
                
        else:        
            if TimeSteppingMethod == "EF":
                integrator = Euler_forward
            elif TimeSteppingMethod == "CN":
                integrator = Crank_Nicholson
            elif TimeSteppingMethod == "RK4":
                integrator = Runge_Kutta4
            for n in range(1, nt):
                un = integrator(un, nx, dx, dt, PhysC.c, du_dt)
                Result[n] = un.copy()
                Var[n] = np.var(un - Result_theory[n], mean = 0).copy()
                    
        return Time, Xaxis, Result, Var
    

#%% Setting paramaters for simulation
#  TODO Let's for now consider a ... m domain split into ... equal parts, such that:
L = 1 # m
nx = 10**2
dx = L / nx # m
# TODO Now we use the fact that we fix the ratio(s) c * Δt / Δx
ratios = np.array([0.25])
# From that we calculate the Δt's
dts = ratios * dx / PhysConstants().c
# TODO Now let's say we want to consider a ... amount timesteps also
nt = 10**3
TotalTime = nt * dts

#%% Running a simulation
#TODO Later this also gotta be for 2 Δt's and also for different derivatives ig
All_Results = {} # We create a dictionary to add all our simulation results to
for j, dt in enumerate(dts):
    for DiffMethod in ["CD"]: # TODO add SP
        for TimeSteppingMethod in ["Theory", "LF"]: #, "EF"]: #, "RK4", "CN"]: # , "AB", "LF",]:
            for Initialisation in ["GaussWave"]: #, "Molenkamp"]:
                Time, Xaxis, Result, Var = Task2_caller(L, nx, TotalTime[j], dt, 
                                                   TimeSteppingMethod, Initialisation,
                                                   DiffMethod)
                All_Results[(TimeSteppingMethod, dt, DiffMethod, Initialisation)] = {
                                                 "Time": Time,
                                                 "Xaxis": Xaxis,
                                                 "Wave": Result,
                                                 "Variances": Var
                                                 }

def u_x_plot(Results, Ng, nt, Init, dtg): #, TSM = False, dt = False, DM = False):
    # dt, nt and DiffMethod are the same variables as before.
    # Ng is the number of graphs the function should make
    # Results should be a dictionary featuring the Results of the simulations
    # in principal this will always be All_Results
    ngs = np.rint(np.linspace(0, nt - 1, num = Ng + 1))
    # ngs = ngs[1:]
    # ngs will be the list of indices of time for which we will plot T(t_(ng); x)
    # This will at least include ng = 0 and ng = Nt (i.e. the first and final steps).
    # Although we cut out the initial situation, as that is non-interesting.
    # The nt - 1 is there because the indices of a list with nt units go from
    # 0 to nt - 1
    for ng in ngs:
        plt.figure()
        for (TimeSteppingMethod, dt, DiffMethod, Initialisation), Result in Results.items():
            if Initialisation == Init and dt == dtg:
                Xaxis = Result["Xaxis"]
                us = Result["Wave"]
                tg = Time[int(ng)]
                ugs = us[int(ng)]
                plt.plot(Xaxis, ugs, '.', label = f'TSM = {TimeSteppingMethod},'
                                                  f'DM = {DiffMethod}, '
                                                  f'$Δt$ = {dtg:.2e} s')
                plt.title(f'$u(t = {tg:.2e}, x)$, Init = {Initialisation}')
        plt.xlim(0, L)
        plt.ylim(0, )
        plt.grid(PhysConstants().Ag)
        plt.xlabel('$x$ (m)')
        plt.ylabel('$u$')
        plt.legend()
        plt.show()


for dt in dts:
    u_x_plot(All_Results, 5, nt, "GaussWave", dt)


# %% Plotting var

def plot_Var(Results):
    plt.figure()
    for (TimeSteppingMethod, dt, DiffMethod, Initialisation), Result in Results.items():
        Time = Result["Time"]
        Variances = Result["Variances"]
        plt.plot(Time, Variances, label = f'TSM = {TimeSteppingMethod}, $Δt$ = {dt} s, '
                                                     f'Init = {Initialisation}')
        plt.title('Variance of different TSM')
        plt.xlabel("$t$ (s)")
        plt.ylabel("$σ^2$")
        plt.legend()
        plt.grid()
        plt.xlim(0, Time[-1])
    plt.show()
        
plot_Var(All_Results)

#%%
"""
At L = 1, nx = 100, ratio = 0.25, 
        self.y0     = 1       # no-signal value #??? Why is this here?
        self.c      = 1       # advection velocity (m/s)
        self.Ag     = 0.5       # Gaussian wave amplitude ()
        self.sigmag = 0.2        # Gaussian wave width (m)
        self.Anot   = 0.5       # Molenkamp triangle height ()
        self.W      = 0.2       # Molenkamp triangle width (m)
Below all for GaussWave:
CN doesn't seem to work well currently. Does some weird stuff where the wave
reappears. Weirdly the variance plateaus about halfway through tho.
RK4 does not work at given parameters. Also doesn't at smaller ratios
LF explodes at the low ratios I tried for RK4.
I think I need to reeavluate the boundary conditions, as now by default both 
ends have a constant value of zero, but that obvioulsy fucks with it all...
"""