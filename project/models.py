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
def PreyPredator_Jac(t:float, x:np.ndarray|list|Tuple[float, float], params:Tuple) -> np.ndarray:
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
    return np.array([x1_,x2_]).reshape(-1,1)

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
Cin = np.array([1.6/2, 2.4/2,600])      # Assume inlet concentration = initial for now

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

    return np.array(dCdt).reshape(-1,1)

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

    r = k * CA * CB    
    k = k0 * np.exp(-Ea_over_R / T_state)

    CA = CA_in + (1/beta)*(T0 - T_state)
    CB = CB_in + (2/beta)*(T0 - T_state)

    # Derivatives
    dk_dT = (Ea_over_R / T_state**2) * k
    dCA_dT = -1/beta
    dCB_dT = -2/beta

    # Full derivative using product rule
    dr_dT = dk_dT * CA * CB + k * dCA_dT * CB + k * CA * dCB_dT

    # return Jacobian, bc 1D simply derivative
    return np.array([
        [-F/V + beta * dr_dT]
    ])

def CSTR1_Fun_Jac(t:float, T_state:float, params:CSTR1_param_type=CSTR1_PARAMS) -> Tuple[float, np.ndarray]:
    return CSTR1(t, T_state, params), CSTR1_Jac(t, T_state, params)



### PFR Plug-Flow-Reactor

