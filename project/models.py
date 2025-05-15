from typing import Tuple
import numpy as np

# Prey predator function valuation; if 4 parameters often called Lotka-Volterra
def PreyPredator(t:float, x:np.ndarray|list|Tuple[float, float], params:Tuple[float, float]=(1, 0.8)) -> np.ndarray:
    a, b = params
    x1, x2 = x
    dx1 = a * (1 - x2) * x1
    dx2 = -b * (1 - x1) * x2
    return np.array([dx1, dx2])

# Jacobian of prey predator evaluated at x(t)
def PreyPredator_Jac(t:float, x:np.ndarray|list|Tuple[float, float], params:Tuple=(1, 0.8)) -> np.ndarray:
    x1, x2 = x
    a, b = params
    dx1x1 = a * (1 - x2) 
    dx1x2 = -a * x1
    dx2x1 = b * x2
    dx2x2 = -b * (1 - x1)
    return np.array([[dx1x1, dx1x2], [dx2x1, dx2x2]])

# Prey predator function and jacobian
def PreyPredator_Fun_Jac(t:float, x:float|np.ndarray|list, params:Tuple[float, float]=(1, 0.8)) ->Tuple[np.ndarray, np.ndarray]:
    return PreyPredator(t, x, params), PreyPredator_Jac(t, x, params)


### Lotka Volterra - basically Predator Prey with 4 parameters
y0 = [10.0, 5.0] # predator and prey concentration
def LotkaVolterra(t:float, y:np.ndarray|list=y0, params:Tuple[float,float,float,float]=(1.1, 0.4, 0.1, 0.4)):
    alpha, beta, delta, gamma = params
    x, y_ = y
    return np.array([
        alpha * x - beta * x * y_,
        delta * x * y_ - gamma * y_
    ])

def LotkaVolterra_Jac(t:float, y:np.ndarray|list=y0, params:Tuple[float,float,float,float]=(1.1, 0.4, 0.1, 0.4)):
    alpha, beta, delta, gamma = params
    x, y_ = y
    return np.array([
        [alpha - beta * y_, -beta * x],
        [delta * y_, delta * x - gamma]
    ])


# Van Der Pol Oscillator function
def VanDerPol(t:float, x:np.ndarray|list|Tuple[float, float], params:Tuple[float]=(10)) -> np.ndarray:
    mu = params
    x1_ = x[1]
    x2_ = mu*(1-x[0]**2)*x[1]-x[0]
    return np.array([x1_,x2_])

def VanDerPol_Jac(t:float, x:np.ndarray|list|Tuple[float, float], params:Tuple[float]=(10)) -> np.ndarray:
    mu = params
    x1, x2 = x
    return np.array([[0, 1],[-2 * mu * x1 * x2 - 1, mu * (1 - x1**2)]])

def VanDerPol_Fun_Jac(t:float, x:np.ndarray|list|Tuple[float, float], params:Tuple[float]=(10)) -> Tuple[np.ndarray, np.ndarray]:
    return VanDerPol(t,x,params), VanDerPol_Jac(t,x,params)



### CSTR CHEMICAL REACTOR

# constants
deltaHr = -560               # kJ/mol
rho = 1.0                    # kg/L
cp = 4.186                   # kJ/(kg*K)
Ea_over_R = 8500             # K
k0 = np.exp(24.6)            # L/(mol*s)
V = 0.105                    # L (from Wahlgreen 2020)
F = 0.1                      # L/s  (assumed constant flowrate, realistic)

# initialisations
C0 = np.array([3.4, 3.7, 200])          # Initial condition
Cin = C0 + np.array([0.03, -0.04, -10])     # Assume inlet concentration = initial for now

CSTR3_PARAMS = (deltaHr, rho, cp, Ea_over_R, k0, V, F, Cin)
CSTR3_param_type = Tuple[float, float, float, float, float, float, float, np.ndarray]

def CSTR3(t:float, C_states:np.ndarray|list|Tuple[float, float, float]=C0, params:CSTR3_param_type=CSTR3_PARAMS) -> np.ndarray:
    deltaHr, rho, cp, Ea_over_R, k0, V, F, Cin = params

    # robustness
    T = max(C_states[2], 1e-3)  # prevent underflow
    T = max(T, 1e-3)            # add temperature lower bound to avoid 0 division error

    # State variables
    CA, CB, T = C_states

    # Arrhenius law
    k = k0 * np.exp(-Ea_over_R/T)

    # Reaction model
    r = k * CA * CB
    beta = -deltaHr / (rho * cp)
    v = np.array([-1, -2, beta])  # stoichiometric vector

    # Reaction term
    R = v * r

    # CSTR dynamics
    dCdt = (Cin - C_states) * F / V + R

    return np.array(dCdt)


def CSTR3_Jac(t: float, C_states: np.ndarray|list=C0, params:CSTR3_param_type=CSTR3_PARAMS) -> np.ndarray:
    deltaHr, rho, cp, Ea_over_R, k0, V, F, Cin = params

    # State variables
    CA, CB, T = C_states

    # Safety check (avoid division by zero or overflow)
    if T < 1e-5:
        raise ValueError("Temperature too low — possible singularity in Jacobian.")

    # Arrhenius law
    k = k0 * np.exp(-Ea_over_R / T)
    r = k * CA * CB
    beta = -deltaHr / (rho * cp)

    # Derivatives
    dk_dT = (Ea_over_R / T**2) * k
    dr_dCA = k * CB
    dr_dCB = k * CA
    #dr_dT  = -Ea_over_R / T**2 * r  # product rule: d(k CA CB)/dT
    dr_dT = dk_dT * CA * CB

    # Partial derivatives of R = v * r
    v = np.array([-1, -2, beta])

    # Construct Jacobian
    J = np.empty((3, 3))

    # d(CA)/dt
    J[0, 0] = -F/V + v[0] * dr_dCA
    J[0, 1] =        v[0] * dr_dCB
    J[0, 2] =        v[0] * dr_dT

    # d(CB)/dt
    J[1, 0] =        v[1] * dr_dCA
    J[1, 1] = -F/V + v[1] * dr_dCB
    J[1, 2] =        v[1] * dr_dT

    # d(T)/dt
    J[2, 0] =        v[2] * dr_dCA
    J[2, 1] =        v[2] * dr_dCB
    J[2, 2] = -F/V + v[2] * dr_dT

    return J


def CSTR3_Fun_Jac(t:float, C_states:np.ndarray|list|Tuple[float, float, float]=C0, params:CSTR3_param_type=CSTR3_PARAMS) -> Tuple[np.ndarray, np.ndarray]:
    return CSTR3(t, C_states, params), CSTR3_Jac(t, C_states, params)



# constants; same as CSTR3 plus... [Wahlgreen et al. (2020)]
CA_in = 1.6/2                # mol/L
CB_in = 2.4/2                # mol/L

# initial conditions
T0 = 273.15                 # initial reactor temperature
Tin = 273.15                # inlet feed temperature

CSTR1_PARAMS = (deltaHr, rho, cp, Ea_over_R, k0, V, F, CA_in, CB_in, T0, Tin)
CSTR1_param_type = Tuple[float, float, float, float, float, float, float, float, float, float, float]

def CSTR1(t:float, T_state:float, params:CSTR1_param_type=CSTR1_PARAMS) -> float:
    if type(T_state)==list: 
        T_state = T_state[0]
    deltaHr, rho, cp, Ea_over_R, k0, V, F, CA_in, CB_in, T0, Tin = params
    beta = -deltaHr / (rho * cp)

    CA = CA_in + 1/beta * (T0-T_state)
    CB = CB_in + 2/beta * (T0-T_state)

    k = k0 * np.exp(-Ea_over_R/T_state)
    r = k * CA * CB

    dTdt = (Tin - T_state) * F/V + beta * r

    return dTdt

def CSTR1_Jac(t:float, T_state:float, params:CSTR1_param_type=CSTR1_PARAMS) -> np.ndarray:
    deltaHr, rho, cp, Ea_over_R, k0, V, F, CA_in, CB_in, T0, Tin = params
    beta = -deltaHr / (rho * cp)

    CA = CA_in + (1/beta)*(T0 - T_state)
    CB = CB_in + (2/beta)*(T0 - T_state)

    k = k0 * np.exp(-Ea_over_R / T_state)
    r = k * CA * CB    

    # Derivatives
    dk_dT = (Ea_over_R / T_state**2) * k
    dCA_dT = -1/beta
    dCB_dT = -2/beta

    # Full derivative using product rule
    dr_dT = dk_dT * CA * CB + k * dCA_dT * CB + k * CA * dCB_dT

    # return Jacobian, bc 1D simply derivative
    return np.array([-F/V + beta * dr_dT])

def CSTR1_Fun_Jac(t:float, T_state:float, params:CSTR1_param_type=CSTR1_PARAMS) -> Tuple[float, np.ndarray]:
    return CSTR1(t, T_state, params), CSTR1_Jac(t, T_state, params)



### PFR Plug-Flow-Reactor

# Initial conditions
Nz, L, v, k, DA, DB, DT, uk = 10, 10, 3, 4, 0.1, 0.1, 0.1, 3
CA0, CB0, T0 = np.full(Nz, DA), np.full(Nz, DB), np.full(Nz, 200)
C0 = np.concatenate([CA0, CB0, T0])

n, dz, Cin = Nz, L/Nz, (1.6, 2.4, 350)
PFR3_PARAMS = (n, dz, v, DA, DB, DT, k, Cin)

def PFR3(t:float, C:np.ndarray|list=C0, params:Tuple[float,float,float,float,float,float,float, Tuple[float,float,float]]=PFR3_PARAMS):
    n, dz, v, DA, DB, DT, k, Cin = params
    CAin, CBin, Tin = Cin

    CA = C[0:n]
    CB = C[n:2*n]
    T = C[2*n:3*n]
    
    deltaHr = -560               
    rho = 1.0                   
    cp = 4.186                  
    Ea_over_R = 8500            
    k0 = np.exp(24.6)           
    V = 0.105                   
    F = 250      
    beta = -deltaHr / (rho * cp)

    k = k0 * np.exp(-Ea_over_R / T)
    r = k * CA * CB
    
    # Convection at finite volume interfaces
    NconvA = np.zeros(n+1)
    NconvA[0] = v * CAin
    NconvA[1:n+1] = v * CA[0:n]
    
    NconvB = np.zeros(n+1)
    NconvB[0] = v * CBin
    NconvB[1:n+1] = v * CB[0:n]

    NconvT = np.zeros(n+1)
    NconvT[0] = v * Tin
    NconvT[1:n+1] = v * T[0:n]
    

    # Diffusion at finite volume interfaces
    JA = np.zeros(n+1)
    JA[1:n] = (-DA/dz) * (CA[1:n] - CA[0:n-1])

    JB = np.zeros(n+1)
    JB[1:n] = (-DB/dz) * (CB[1:n] - CB[0:n-1])

    JT = np.zeros(n+1)
    JT[1:n] = (-DT/dz) * (T[1:n] - T[0:n-1])
    
    # Flux = convection + diffusion
    NA = NconvA + JA
    NB = NconvB + JB
    NT = NconvT + JT
    
    # Reaction and production rates in finite volumes
    r = k * CA * CB
    RA = -r
    RB = -2*r
    RT = beta*r
    
    # Differential Equations (mass balances at finite volumes)
    CAdot = (NA[1:n+1] - NA[0:n])/(-dz) + RA
    CBdot = (NA[1:n+1] - NB[0:n])/(-dz) + RB
    Tdot = (NA[1:n+1] - NT[0:n])/(-dz) + RT
    
    return np.vstack([CAdot, CBdot, Tdot]).flatten()


### Lorentz-Attractor
sigma, beta, rho = 10.0, 8/3, 28.0
y0 = [1, 1, 1]
LORENTZ_PARAMS = (sigma, beta, rho)
def LorentzAttractor(t, y:np.ndarray|list=y0, params:Tuple[float, float, float]=LORENTZ_PARAMS):
  x, y, z = y
  sigma, beta, rho = params
  dxdt = sigma * (y - x)
  dydt = x * (rho - z) - y
  dzdt = x * y - beta * z
  return np.array([dxdt, dydt, dzdt])

def LorentzAttractor_Jac(t:float, y:np.ndarray|list, params:Tuple[float, float, float]=LORENTZ_PARAMS):
    sigma, beta, rho = params
    x, y, z = y
    return np.array([
        [sigma, sigma, 0], 
        [rho, -1, x], 
        [y, x, -beta]
    ])

### PFR1 - PipeAdvection Diffusion - WIP
# example initial conditions
Nz = 10
dz = 1
v = 1
DA = 0.1
k = 0.1
p = {"Nz": Nz, "dz": dz, "v": v, "DA": DA, "k": k}
t0 = 0
tN = 10
N = 100
x0 = np.zeros(Nz)
x0[0] = 1
u = 1

def PipeAdvectionDiffusionReaction1(t, x, u, p):
    cA = x
    cAin = u

    n = p["Nz"]
    dz = p["dz"]
    v = p["v"]
    DA = p["DA"]
    k = p["k"]

    # convection at finite volume interfaces
    NconvA = np.zeros(n+1)
    NconvA[0] = v * cAin
    NconvA[1:n+1] = v * cA[0:n]

    # diffusion at finite volume interfaces
    JA = np.zeros(n+1)
    JA[1:n] = (-DA / dz) * (cA[1:n] - cA[0:n-1])

    # flux = convection + diffusion
    NA = NconvA + JA

    # reaction and production rates in finite volumes
    r = k * cA
    RA = -r

    # Differential Equations (mass balances at finite volumes)
    cAdot = (NA[1:n+1] - NA[0:n]) / (-dz) + RA
    xdot = cAdot

    return xdot


### Brusselator
# Initial conditions
A = 2.0  # Reaction rate parameter A for substance u
B = 3.0  # Reaction rate parameter B for substance v&u

# Initial concentrations of u and v at time t=0
u_0 = A
v_0 = B / (A - 1)
y0 = [u_0, v_0]

BRUSSELATOR_PARAMS = (A, B)
def Brusselator(t:float, y:np.ndarray|list=y0, params:Tuple[float, float]=BRUSSELATOR_PARAMS):
    u, v = y
    A, B = params
    du_dt = A - (B + 1) * u + u**2 * v  # Rate of change for u
    dv_dt = B * u - u**2 * v             # Rate of change for v
    return np.array([du_dt, dv_dt])

def Brusselator_Jac(t:float, y:np.ndarray|list=y0, params:Tuple[float, float]=BRUSSELATOR_PARAMS):
    u, v = y
    A, B = params
    return np.array([
        [A-(B+1)+2*u*v, u**2],
        [B-2*u*v, u**2]
    ])


### Robertson Chemical Reaction
# Initial Conditions
k1, k2, k3, k4 = 0.1, 0.2, 0.3, 0.05
ROBERTSON_PARAMS = k1, k2, k3, k4
y0 = [1.0, 0.1, 0.1, 0.1]

def Robertson(t:float, y:np.ndarray|list=y0, params:Tuple[float,float,float,float]=ROBERTSON_PARAMS):
    """
    Robertson chemical reaction model (Hovorka model).

    Args:
        y: A list or array containing the concentrations of the four species:
           [A, B, C, D].
        t: The current time.
        k1: Rate constant for the first reaction (A -> 2B)
        k2: Rate constant for the second reaction (2B -> C)
        k3: Rate constant for the third reaction (C -> 2D)
        k4: Rate constant for the fourth reaction (2D -> B)

    Returns:
        A list containing the derivatives of the concentrations [dA/dt, dB/dt, dC/dt, dD/dt].
    """
    k1, k2, k3, k4 = params
    A, B, C, D = y

    dadt = -k1 * A
    dbdt = k1 * A - 2 * k2 * B + k4 * 2 * D
    dcdt = 2 * k2 * B - k3 * C
    dddt = k3 * C - k4 * 2 * D

    return np.array([dadt, dbdt, dcdt, dddt])

def Robertson_Jac(t:float, y:np.ndarray|list=y0, params:Tuple[float,float,float,float]=ROBERTSON_PARAMS):
    k1, k2, k3, k4 = params
    A, B, C, D = y
    return np.array([
        [-k1, 0, 0, 0],
        [k1, -2*k2, 0, 2*k4],
        [0, 2*k2, -k3, 0],
        [0, 0, k3, -2*k4]
    ])


### Hovorka Glucose-Insulin Model

# Model parameters (adjust as needed)
BW = 70  # Body weight (kg)
Kc = 0.01  # Glucose conversion rate (kg/h)
Km = 0.5  # Michaelis constant (g/L)
Vm = 0.1  # Maximum conversion rate (g/L/h)
tau_g = 0.1  # Glucose appearance rate constant (h^-1)
tau_i = 0.2  # Insulin clearance rate constant (h^-1)
tau_p = 0.1  # Pancreas beta cell production constant (h^-1)
HOVORKA_PARAMS = (BW, Kc, Km, Vm, tau_g, tau_i, tau_p)
y0 = [5.0, 50.0, 1.0, 1.0, 1.0, 10.0, 10.0, 100.0, 10.0, 50.0]

def Hovorka(t:float, y:np.ndarray|list, params:Tuple[float,float,float,float,float,float,float]=HOVORKA_PARAMS):
    """
    Hovorka model for glucose-insulin regulation.

    Args:
        y: A list or array containing the concentrations of the 10 variables:
           [Glucose, Insulin, LiverGlucose, MuscleGlucose, BrainGlucose,
            LiverGlycogen, MuscleGlycogen, PancreasBetaCell,
            PancreasAlphaCell, InsulinSecretionRate].
        t: The current time.
        BW: Body weight (kg).
        Kc: Glucose conversion rate (kg/h)
        Km: Michaelis constant for glucose conversion (g/L)
        Vm: Maximum conversion rate (g/L/h)
        tau_g: Glucose appearance rate constant (h^-1)
        tau_i: Insulin clearance rate constant (h^-1)
        tau_p: Pancreas beta cell production constant (h^-1)

    Returns:
        A list containing the derivatives of the 10 variables [dGlucose/dt, dInsulin/dt, ...].
    """

    # Unpack variables
    BW, Kc, Km, Vm, tau_g, tau_i, tau_p = params
    G, I, GL, GM, GB, LGLY, MGLY, BETA, ALPHA, ISR = y

    # Model equations
    dGdt = -tau_g * G + (Kc / (GL + GL * (GL / Km)) + Kc / (GM + GM * (GM / Km)) + Kc / (GB + GB * (GB/ Km)))
    dIdt = -tau_i * I + ISR
    dGLdt = -GL + Kc / (GL + GL * (GL / Km))  # Simplified Liver glucose equation.  Adjust as needed.
    dGMdt = -GM + Kc / (GM + GM * (GM / Km)) # Simplified Muscle glucose equation. Adjust as needed.
    dGBdt = -GB + Kc / (GB + GB * (GB / Km)) # Simplified Brain glucose equation. Adjust as needed.
    dLGLYdt = -LGLY
    dMGLYdt = -MGLY
    dBETAdt = ISR
    dALPhadt = -ISR
    dISRdt = ISR*(1-ISR/100) # Simplified Insulin secretion rate.  Adjust as needed.

    return np.array([dGdt, dIdt, dGLdt, dGMdt, dGBdt, dLGLYdt, dMGLYdt, dBETAdt, dALPhadt, dISRdt])

def Hovorka_Jac(t:float, y:np.ndarray|list=y0, params:Tuple[float,float,float,float,float,float,float]=HOVORKA_PARAMS):
    """
    Analytical Jacobian of the simplified Hovorka model.
    
    Args:
        y: State vector of length 10.
        t: Time (unused, but included for compatibility).
        BW, Kc, Km, Vm, tau_g, tau_i, tau_p: Model parameters.
        
    Returns:
        J: 10x10 numpy array, the Jacobian matrix.
    """
    G, I, GL, GM, GB, LGLY, MGLY, BETA, ALPHA, ISR = y
    BW, Kc, Km, Vm, tau_g, tau_i, tau_p = params
    J = np.zeros((10, 10))

    ### dGdt = -tau_g * G + f(GL, GM, GB)
    # Partial derivatives of f(GL, GM, GB) w.r.t. GL, GM, GB
    # Common derivative pattern for GL, GM, GB
    def df_dX(X):  
        denom = X + X * (X / Km)
        return -Kc * (1 + 2 * X / Km) / (denom ** 2)

    J[0, 0] = -tau_g                         # ∂(dGdt)/∂G
    J[0, 2] = df_dX(GL)                      # ∂(dGdt)/∂GL
    J[0, 3] = df_dX(GM)                      # ∂(dGdt)/∂GM
    J[0, 4] = df_dX(GB)                      # ∂(dGdt)/∂GB

    ### dIdt = -tau_i * I + ISR
    J[1, 1] = -tau_i                         # ∂(dIdt)/∂I
    J[1, 9] = 1                              # ∂(dIdt)/∂ISR

    ### dGLdt = -GL + Kc / (GL + GL*(GL/Km))
    denom_GL = GL + GL * (GL / Km)
    dGL_rhs = -1 + df_dX(GL)
    J[2, 2] = dGL_rhs                        # ∂(dGLdt)/∂GL

    ### dGMdt = -GM + Kc / (GM + GM*(GM/Km))
    denom_GM = GM + GM * (GM / Km)
    dGM_rhs = -1 + df_dX(GM)
    J[3, 3] = dGM_rhs                        # ∂(dGMdt)/∂GM

    ### dGBdt = -GB + Kc / (GB + GB*(GB/Km))
    denom_GB = GB + GB * (GB / Km)
    dGB_rhs = -1 + df_dX(GB)
    J[4, 4] = dGB_rhs                        # ∂(dGBdt)/∂GB

    ### dLGLYdt = -LGLY
    J[5, 5] = -1                             # ∂(dLGLYdt)/∂LGLY

    ### dMGLYdt = -MGLY
    J[6, 6] = -1                             # ∂(dMGLYdt)/∂MGLY

    ### dBETAdt = ISR
    J[7, 9] = 1                              # ∂(dBETAdt)/∂ISR

    ### dALPHAdt = -ISR
    J[8, 9] = -1                             # ∂(dALPHAdt)/∂ISR

    ### dISRdt = ISR*(1 - ISR/100)
    J[9, 9] = (1 - 2 * ISR / 100)            # ∂(dISRdt)/∂ISR

    return J

# enhanced version of the Hovorka model; pancreatic feedback, saturation effects
k_syn = 0.1     # Glycogen synthesis rate (h^-1)
k_deg = 0.05    # Glycogen degradation rate (h^-1)
Gb = 0.8        # Basal glucose (g/L)
Ib = 0.3        # Basal insulin (mU/L)
HOVORKA_PARAMS_ = BW, Kc, Km, Vm, tau_g, tau_i, tau_p, k_syn, k_deg, Gb, Ib

def Hovorka_(t:float, y:np.ndarray|list=y0, params:Tuple[float,float,float,float,float,float,float,float,float,float,float]=HOVORKA_PARAMS_):
    """
    Enhanced Hovorka model with more physiological realism.

    Args:
        y: State vector of 10 variables:
           [G, I, GL, GM, GB, LGLY, MGLY, BETA, ALPHA, ISR]
        t: Time (unused).
        BW: Body weight (kg)
        Kc: Glucose conversion rate (kg/h)
        Km: Michaelis constant for glucose transport (g/L)
        Vm: Max glucose uptake rate (g/L/h)
        tau_g: Glucose clearance rate (h^-1)
        tau_i: Insulin clearance rate (h^-1)
        tau_p: Beta cell production rate (h^-1)
        k_syn: Glycogen synthesis rate (h^-1)
        k_deg: Glycogen degradation rate (h^-1)
        Gb: Basal glucose (g/L)
        Ib: Basal insulin (mU/L)

    Returns:
        dydt: List of 10 derivatives.
    """

    # Unpack variables
    G, I, GL, GM, GB, LGLY, MGLY, BETA, ALPHA, ISR = y
    BW, Kc, Km, Vm, tau_g, tau_i, tau_p, k_syn, k_deg, Gb, Ib = params

    # Michaelis-Menten terms for glucose uptake
    def glucose_uptake(X): return Kc / (X + X * (X / Km))

    # Pancreatic insulin secretion (Hill-type function)
    ISR = Vm * (G**2 / (G**2 + Km**2)) * BETA

    # Glucose dynamics (influenced by insulin-enhanced uptake)
    U_id = Vm * I / (Km + I + 1e-8)  # insulin-dependent uptake
    HGP = max(0, Gb - G) * (1 - I / (Ib + 1e-8))  # hepatic glucose production suppressed by insulin

    dGdt = -tau_g * G - U_id + HGP + glucose_uptake(GL) + glucose_uptake(GM) + glucose_uptake(GB)
    dIdt = -tau_i * I + ISR
    dGLdt = -glucose_uptake(GL) + k_deg * LGLY
    dGMdt = -glucose_uptake(GM) + k_deg * MGLY
    dGBdt = -glucose_uptake(GB)  # brain glucose uptake is mostly insulin-independent

    dLGLYdt = k_syn * GL - k_deg * LGLY
    dMGLYdt = k_syn * GM - k_deg * MGLY

    dBETAdt = tau_p * (G - Gb)  # Glucose stimulates β-cell growth
    dALPHAdt = -tau_p * (G - Gb)  # Glucose suppresses α-cells
    dISRdt = (ISR * (1 - ISR / 100))  # bounded ISR (simple model of saturation)

    return np.array([dGdt, dIdt, dGLdt, dGMdt, dGBdt, dLGLYdt, dMGLYdt, dBETAdt, dALPHAdt, dISRdt])

def Hovorka_Jac_(t:float, y:np.ndarray|list=y0, params:Tuple[float,float,float,float,float,float,float,float,float,float,float]=HOVORKA_PARAMS_):
    BW, Kc, Km, Vm, tau_g, tau_i, tau_p, k_syn, k_deg, Gb, Ib = params
    G, I, GL, GM, GB, LGLY, MGLY, BETA, ALPHA, ISR = y
    J = np.zeros((10, 10))

    # Helper functions
    def dUptake_dX(X):  # Derivative of glucose uptake
        denom = (X + X * (X / Km))
        return -Kc * (1 + 2 * X / Km) / (denom ** 2)

    def dU_id_dI(I):  # derivative of insulin-dependent uptake
        return Vm * Km / (Km + I + 1e-8) ** 2

    def dHGP_dG(G):  # ∂HGP/∂G
        return -1 if G < Gb else 0

    def dHGP_dI(I):  # ∂HGP/∂I
        return -max(0, Gb - G) / (Ib + 1e-8)

    # ISR = Vm * (G^2 / (G^2 + Km^2)) * BETA
    dISR_dG = Vm * (2 * G * Km**2 * BETA) / (G**2 + Km**2)**2
    dISR_dBETA = Vm * (G**2 / (G**2 + Km**2))

    ### dG/dt
    J[0, 0] = -tau_g - dHGP_dG(G)
    J[0, 1] = -dU_id_dI(I) + dHGP_dI(I)
    J[0, 2] = dUptake_dX(GL)
    J[0, 3] = dUptake_dX(GM)
    J[0, 4] = dUptake_dX(GB)

    ### dI/dt
    J[1, 0] = dISR_dG
    J[1, 7] = dISR_dBETA
    J[1, 1] = -tau_i

    ### dGL/dt
    J[2, 2] = -dUptake_dX(GL)
    J[2, 5] = k_deg

    ### dGM/dt
    J[3, 3] = -dUptake_dX(GM)
    J[3, 6] = k_deg

    ### dGB/dt
    J[4, 4] = -dUptake_dX(GB)

    ### dLGLYdt
    J[5, 2] = k_syn
    J[5, 5] = -k_deg

    ### dMGLYdt
    J[6, 3] = k_syn
    J[6, 6] = -k_deg

    ### dBETAdt
    J[7, 0] = tau_p

    ### dALPHAdt
    J[8, 0] = -tau_p

    ### dISRdt
    J[9, 0] = dISR_dG * (1 - ISR / 100)
    J[9, 7] = dISR_dBETA * (1 - ISR / 100)
    J[9, 9] = 1 - 2 * ISR / 100

    return J