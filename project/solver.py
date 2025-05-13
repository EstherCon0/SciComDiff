import numpy as np
from typing import Callable, Tuple
#from utils import StdMultivariateWienerProcess

def dopri54_step(f:Callable, t:float, x:np.ndarray | float, h:float=1e-2, *args) -> Tuple[float, float]:

    # Runge-Kutta Explicit Order 4

    # Coefficients for Dormand-Prince method
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    b1p, b2p, b3p, b4p, b5p, b6p, b7p = 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40

    # Calculate the stages
    k1 = h * f(t, x, *args)
    k2 = h * f(t + c2*h, x + a21*k1, *args)
    k3 = h * f(t + c3*h, x + a31*k1 + a32*k2, *args)
    k4 = h * f(t + c4*h, x + a41*k1 + a42*k2 + a43*k3, *args)
    k5 = h * f(t + c5*h, x + a51*k1 + a52*k2 + a53*k3 + a54*k4, *args)
    k6 = h * f(t + c6*h, x + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5, *args)
    k7 = h * f(t + c7*h, x + a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6, *args)

    # Compute the next value
    x_next = x + b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6 + b7*k7

    # Estimate error with embedded method
    x_err = abs(b1p*k1 + b2p*k2 + b3p*k3 + b4p*k4 + b5p*k5 + b6p*k6 + b7p*k7 - x_next)

    return x_next, x_err

def adaptive_dopri54(f:Callable, x0:np.ndarray | float, t0:float=0.0, t_end:float=10.0, h:float=0.1, eps:float=0.8, max_iter:int=1e5, *args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Adaptive Runge-Kutta 4(5) integration
    t = t0
    x = np.array(x0)
    times = [t]
    results = [x.copy()]
    err = [0]
    p = 4                       # integrator order
    k_p = 0.4/(p+1)
    k_i = 0.3/(p+1)
    r_acc = 1
    iter_ = 1

    while t < t_end:
        if t + h > t_end:  # Adjust last step to reach exactly t_end
            h = t_end - t
        if iter_>max_iter: raise AssertionError("Max iterations reached")

        x_next, err_est = dopri54_step(f, t, x, h, *args)

        # Estimate the error and adjust step size
        r = np.linalg.norm(err_est, ord=2)
        err.extend(err_est)
        
        if r <= 1.0:
            t += h
            x = x_next
            times.append(t)
            results.append(x.copy())
            h *= ((eps/r)**k_i)*((r_acc/r)**k_p)
            r_acc = r
        
        else:
            h *= (eps/r)**(1/iter_)
        
        iter_+=1
        if iter_%500==0: print(f"Iteration {iter_}, error_norm: {r}")

    print(f"Total of {iter_} iterations.")
    return np.array(times), np.array(results), np.array(err)

def adaptive_rk45(f:Callable, x0:np.ndarray|float, t0:float=0.0, t_end:float=10.0, h:float=0.1, tol:float=1e-2, eps:float=0.9, *args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Adaptive Runge-Kutta 4(5) / DOPRI(5)4
    t = t0
    x = np.array(x0)
    times = [t]
    results = [x.copy()]
    err = [tol]
    iter_ = 1

    while t < t_end:
        if t + h > t_end:  # Adjust last step to reach exactly t_end
            h = t_end - t

        x_next, err_est = dopri54_step(f, t, x, h)

        # Estimate the error and adjust step size
        scale = np.maximum(np.abs(x), np.abs(x_next)) + tol
        error_norm = np.linalg.norm(err_est / scale) / np.sqrt(len(x))
        err.extend(err_est)
        
        if error_norm <= 1.0:
            t += h
            x = x_next
            times.append(t)
            results.append(x.copy())
            
        # Increase step size for the next iteration
        if error_norm > 0:  # Avoid division by zero
            h *= min(5, max(0.8, eps * (1 / error_norm) ** 0.2))
        
        else:
            # Decrease step size and try again
            h *= max(0.1, eps * (1 / error_norm) ** 0.25)
        
        iter_+=1
        if iter_%200==0: print(f"Iteration {iter_}, error_norm: {error_norm}")

    return np.array(times), np.array(results), np.array(err)


def ExplicitEulerFixedStep(fun:Callable, x0:np.ndarray | float, t0:float=0, tN:float=10, N:int=100, *args) -> Tuple[np.ndarray, np.ndarray]:
    dt = (tN - t0) / N
    nx = len(x0)

    X = np.zeros((N+1, nx))
    T = np.zeros(N+1)

    # Euler's Explicit Method
    T[0] = t0
    X[0, :] = x0

    for k in range(N):
        f = np.array(fun(T[k], X[k, :], *args))
        T[k+1] = T[k] + dt
        X[k+1, :] = X[k, :] + f * dt

    return T, X

def ExplicitEulerAdaptiveStep(fun:Callable, tspan:list|Tuple[float, float], x0:float|np.ndarray, h0:float=0.1, abstol:float=1e-3, reltol:float=1e-3, *args) -> Tuple[np.ndarray, np.ndarray]:
    # Error controller parameters
    epstol = 0.8  # Safety factor
    facmin = 0.1  # Maximum decrease factor
    facmax = 5.0  # Maximum increase factor

    # Integration interval
    t0, tf = tspan

    # Initial conditions
    t = t0
    h = h0
    x = np.array(x0, dtype=float)  # Ensure x is a NumPy array

    # Counters
    nfun = 0
    naccept = 0
    nreject = 0

    # Output storage
    T = np.array([t])  # Ensure T is a 1D NumPy array
    X = np.array([x])  # Ensure X is a 2D NumPy array

    # Algorithm
    while t < tf:
        if t + h > tf:
            h = tf - t

        f = np.array(fun(t, x, *args), dtype=float)  # Ensure f is a NumPy array

        AcceptStep = False

        while not AcceptStep:
            x1 = x + h * f
            hm = 0.5 * h

            tm = t + hm
            xm = x + hm * f

            fm = np.array(fun(tm, xm, *args), dtype=float)  # Ensure fm is a NumPy array
            nfun += 3
            x1hat = xm + hm * fm

            # Error estimation
            e = x1hat - x1
            denom = np.maximum(abstol, np.abs(x1hat) * reltol)  # Fix element-wise max
            r = np.max(np.abs(e) / denom)  # Compute max ratio

            AcceptStep = r <= epstol

            if AcceptStep:
                t = t + h
                x = x1hat

                naccept += 1
                T = np.append(T, t)  # Append t to the 1D array
                X = np.vstack([X, x1hat.reshape(1, -1)])  # Ensure correct shape
            else:
                nreject += 1

            h = np.max([facmin, np.min([np.sqrt(epstol / r), facmax])]) * h

    return T, X

### PLEASE UPDATE WHAT IS "dt", "psi"??
def SDENewtonSolver(funJ:Callable, t:float, dt, psi, xinit, tol:float=1e-3, maxit:int=200, *args):
    
    # function needs to return the Jacobian as well
    if len(funJ(t, x, *args))<2:
        raise AttributeError("Please provide a function, that also returns the Jacobian of the system")
    
    I = np.eye(len(xinit))      # Identity matrix size x
    x = xinit                   # Initial guess
    f, J = funJ(t, x, *args)    # Evaluating function for f val and jacobian
    R = x - f * dt - psi        # Residual function
    it = 1                      # Iteration count
    
    # While residual is larger than the tolerance
    while np.linalg.norm(R, np.inf) > tol and it <= maxit:
        # Jacobian of residual
        dRdx = I - J * dt   
        # Change in x
        mdx = np.linalg.solve(dRdx, R)
        # Update x
        x = x - mdx
        # Compute function value and jacobian
        f, J = funJ(t, x, *args)
        # Residual
        R = x - f * dt - psi
        # Iteration count
        it += 1
    #what is this one changing for each iteration? The step 

    return x


def SDEsolverExplicitExplicit(ffun:Callable, gfun:Callable, T:np.ndarray, x0:float|np.ndarray, W:np.ndarray|None=None, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an SDE using an explicit-explicit Euler-Maruyama scheme.

    Parameters:
    ffun : function
        Drift function f(t, x, *args).
    gfun : function
        Diffusion function g(t, x, *args).
    T : ndarray
        Array of time points.
    x0 : ndarray
        Initial condition.
    W : ndarray
        Wiener process increments (same shape as T).
    *args : tuple
        Additional parameters for ffun and gfun.

    Returns:
    X : ndarray
        Solution of the SDE at end
    """
    # Number of timesteps
    N = len(T)
    # Number of dimensions of x
    nx = len(x0)
    # Allocating space for the solution 
    X = np.zeros((nx, N))
    # Storing the initial value in X
    X[:, 0] = x0
    # creating white noise
    if W==None:
        W, _, _ = StdMultivariateWienerProcess(T[-1], N, nx, 1, 42)


    for k in range(N - 1):
        # Timestep
        dt = T[k + 1] - T[k]
        # White noise 
        dW = W[:, k + 1] - W[:, k]
        
        # Evaluating the drift function
        f,_ = ffun(T[k], X[:, k], *args)
        
        # Evaluating the diffusion function
        g = gfun(T[k], X[:, k], *args)
        
        # SDE psi definition
        psi = X[:, k] + g * dW  

        # Explicit step to get Xk+1
        X[:, k + 1] = np.array(psi) + np.array(f) *np.array(dt)
    return T, X

def SDEsolverImplicitExplicit(ffun:Callable, gfun:Callable, T:np.ndarray, x0:float|np.ndarray, W:np.ndarray|None=None, *args):
    tol = 1.0e-8            # Tolerance
    maxit = 100             # Max iterations
    
    N = len(T)              # Time steps
    nx = len(x0)            # Dimensions of solution
    X = np.zeros((nx, N))   # Allocating space for solution over time
    X[:, 0] = x0            # Storing initial x in X

    # creating white noise
    if W==None:
        W, _, _ = StdMultivariateWienerProcess(T[-1], N, nx, 1, 42)
    
    for k in range(N - 1):
        # Evaluating the diffusion term
        g = gfun(T[k], X[:, k], *args)
        # Timestep
        dt = T[k + 1] - T[k]
        
        # Corresponding white noise
        dW = W[:, k + 1] - W[:, k]
        
        # SDE definition for psi
        psi = X[:, k] + g * dW

        # Evaluating the drift term
        f,_ = ffun(T[k], X[:, k], *args)

        # Using the explicit as initial guess
        xinit = np.array(psi) + np.array(f) * np.array(dt)

        # Implicit step
        X[:, k + 1] = SDENewtonSolver(
            ffun, T[k + 1], dt, psi, xinit, tol, maxit, *args
        )
    
    return X