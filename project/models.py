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
C0 = np.array([1.6/2, 2.4/2, 600])      # Initial condition
Cin = np.array([1.6/2, 2.4/2, 600])      # Assume inlet concentration = initial for now

CSTR3_PARAMS = (deltaHr, rho, cp, Ea_over_R, k0, V, F, Cin)
CSTR3_param_type = Tuple[float, float, float, float, float, float, float, np.ndarray]

def CSTR3(t:float, C_states:np.ndarray|list|Tuple[float, float, float]=C0, params:CSTR3_param_type=CSTR3_PARAMS) -> np.ndarray:
    deltaHr, rho, cp, Ea_over_R, k0, V, F, Cin = params

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

def CSTR3_Jac(t:float, C_states:np.ndarray|list|Tuple[float, float, float]=C0, params:CSTR3_param_type=CSTR3_PARAMS) -> np.ndarray:
    deltaHr, rho, cp, Ea_over_R, k0, V, F, Cin = params

    # State variables
    CA, CB, T = C_states

    # Arrhenius law
    k = k0 * np.exp(-Ea_over_R/T)
    r = k * CA * CB
    beta = -deltaHr / (rho * cp)

    # Stoichiometry
    v = np.array([-1, -2, beta])

    # Partial derivatives
    dk_dT = (Ea_over_R / T**2) * k

    # return the Jacobian matrix
    return np.array([
        [-F/V - k*CB,      -k*CA,       -(dk_dT * CA * CB)],
        [-F/V - 2*k*CB,    -F/V - 2*k*CA, -(2 * dk_dT * CA * CB)],
        [beta*k*CB,        beta*k*CA,    beta*(-dk_dT*CA*CB) + (-F/V)]
    ])

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
def Brusselator(t:float, y:np.ndarray|list, params:Tuple[float, float]=BRUSSELATOR_PARAMS):
    u, v = y
    A, B = params
    du_dt = A - (B + 1) * u + u**2 * v  # Rate of change for u
    dv_dt = B * u - u**2 * v             # Rate of change for v
    return np.array([du_dt, dv_dt])

def Brusselator_Jac(t:float, y:np.ndarray|list, params:Tuple[float, float]=BRUSSELATOR_PARAMS):
    u, v = y
    A, B = params
    return np.array([
        [A-(B+1)+2*u*v, u**2],
        [B-2*u*v, u**2]
    ])