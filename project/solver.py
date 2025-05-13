import numpy as np
from typing import Callable, Tuple
from .utils import StdMultivariateWienerProcess

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

    # Estimate error with embedded method / relative error
    x_err = abs(b1p*k1 + b2p*k2 + b3p*k3 + b4p*k4 + b5p*k5 + b6p*k6 + b7p*k7 - x_next)

    return x_next, x_err

class DOPRI54:
    def __init__(self, rtol=1e-6, atol=1e-9, h_init:float=0.01, h_min=1e-6, h_max=1.0, min_factor:float=0.1, max_factor:float=5.0):
        self.rtol = rtol
        self.atol = atol
        self.h = h_init
        self.h_min = h_min
        self.h_max = h_max
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.order = 4
        self.stages = 7

        # Dormand-Prince coefficients (Butcher tableau)
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1]).reshape(-1,1)
        self.A = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])
        self.b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        self.b_hat = np.array([5179/57600, 0, 7571/16695, 393/640,-92097/339200, 187/2100, 1/40])
        self.e = np.array([71/57600, 0, -71/16695, 71/1920, 17253/339200, 22/525, -1/40])

    def step(self, f:Callable, t:float, x:float|np.ndarray, h:float):
        k = []
        for i in range(7):
            xi = x.copy()
            for j in range(i):
                xi += h * self.A[i][j] * k[j]
            k.append(f(t + self.c[i] * h, xi))

        x_next = x + h * np.dot(self.b, k)
        x_err = h * np.dot(self.b - self.b_hat, k)
        return x_next, x_err
    
    def adaptive_solve(self, f:Callable, x0:np.ndarray|float, t0:float=0.0, t_end:float=10.0, h0:float=0.01, safety_factor:float=0.9, control_method:str='limit') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_vals, x_vals, err_vals = [t0], [x0], [self.rtol]
        t, x, h = t0, x0, h0

        if control_method=='limit':
            k_iter = 1
            while t < t_end:
                if t + h > t_end:
                    h = t_end - t
                x_new, err = self.step(f, t, x, h)
                err_vals.append(err)
                err_norm = np.linalg.norm(err / (self.atol + self.rtol * np.abs(x_new)), ord=np.inf)

                # accept step
                if err_norm <= 1:
                    t += h
                    x = x_new
                    t_vals.append(t)
                    x_vals.append(x)
                    h = min(self.h_max, h * min(self.max_factor, safety_factor * (1 / err_norm)**(1/self.order+1)))
                else:
                    h = max(self.h_min, h * max(self.min_factor, safety_factor * (1 / err_norm)**(1/self.order)))
            k_iter+=1
            if k_iter%100==0: print(f"Iteration {k_iter}")

            return np.array(t_vals), np.array(x_vals), np.array(err_vals)
        
        elif control_method=='pi':
            eps, k_i, k_p = 0.8, 0.3/(self.order+1), 0.4/(self.order+1)
            k_iter = 1
            while t < t_end:
                if t + h > t_end:  # Adjust last step to reach exactly t_end
                    h = t_end - t

                x_new, err = dopri54_step(f, t, x, h, *args)

                # Estimate the error and adjust step size
                r = np.linalg.norm(err, ord=2)
                err_vals.extend(err)
                
                # accept step
                if r <= 1.0:
                    t += h
                    x = x_new
                    t_vals.append(t)
                    x_vals.append(x)
                    h *= ((eps/r)**k_i)*((r_acc/r)**k_p)
                    r_acc = r
                
                else:
                    h *= (eps/r)**(1/k_iter)

                k_iter+=1

            return np.array(t_vals), np.array(x_vals), np.array(err_vals)


def adaptive_dopri54(f:Callable, x0:np.ndarray|float, t0:float=0.0, t_end:float=10.0, h0:float=0.1, safety_factor:float=0.9, atol:float=1e-9, rtol:float=1e-6, control_method:str='limit', *args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_vals, x_vals, err_vals = [t0], [x0], [rtol]
    t, x, h = t0, x0, h0
    order = 4

    if control_method=='limit':
        min_factor, max_factor, h_min, h_max = 0.1, 5, 1e-6, 1.0
        k_iter = 1
        while t < t_end:
            if t + h > t_end:
                h = t_end - t
            x_new, err = dopri54_step(f, t, x, h, *args)
            err_vals.append(err)
            err_norm = np.linalg.norm(err / (atol + rtol * np.abs(x_new)), ord=np.inf)

            # accept step
            if err_norm <= 1:
                t += h
                x = x_new
                t_vals.append(t)
                x_vals.append(x)
                h = min(h_max, h * min(max_factor, safety_factor * (1 / err_norm)**(1/order+1)))
            else:
                h = max(h_min, h * max(min_factor, safety_factor * (1 / err_norm)**(1/order)))
        k_iter+=1
        if k_iter%100==0: print(f"Iteration {k_iter}")

        return np.array(t_vals), np.array(x_vals), np.array(err_vals)
    
    elif control_method=='pi':
        eps, k_i, k_p = 0.8, 0.3/(order+1), 0.4/(order+1)
        k_iter = 1
        while t < t_end:
            if t + h > t_end:  # Adjust last step to reach exactly t_end
                h = t_end - t

            x_new, err = dopri54_step(f, t, x, h, *args)

            # Estimate the error and adjust step size
            r = np.linalg.norm(err, ord=2)
            err_vals.extend(err)
            
            # accept step
            if r <= 1.0:
                t += h
                x = x_new
                t_vals.append(t)
                x_vals.append(x)
                h *= ((eps/r)**k_i)*((r_acc/r)**k_p)
                r_acc = r
            
            else:
                h *= (eps/r)**(1/k_iter)

            k_iter+=1

        return np.array(t_vals), np.array(x_vals), np.array(err_vals)


def adaptive_rk45(f:Callable, x0:np.ndarray|float, t0:float=0.0, t_end:float=10.0, h:float=0.1, atol:float=1e-3, rtol:float=1e-2, safety:float=0.9, maxFactor:float=10, minFactor:float=0.2, *args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Adaptive Runge-Kutta 4(5) / DOPRI(5)4
    t, p = t0, 4                                    # initial time-step t0, p order of method
    x = np.array(x0)
    times, results, err = [t], [x.copy()], [atol]   # storage lists
    iter_, accepted = 1, False

    while t < t_end:
        if t + h > t_end:  # Adjust last step to reach exactly t_end
            h = t_end - t

        x_next, err_est = dopri54_step(f, t, x, h)

        # Estimate the error and adjust step size
        scale = np.maximum(np.abs(x), np.abs(x_next))*rtol + atol
        error_norm = np.linalg.norm(err_est*h / scale) / np.sqrt(len(x))
        err.extend(err_est)
        
        # accept step
        if error_norm < 1:
            t += h
            x = x_next
            times.append(t)
            results.append(x.copy())

            if error_norm==0: factor = maxFactor
            else: factor = min(maxFactor, safety*error_norm**(1/p+1))
            
            if accepted==False: factor = min(1, factor)
            h *= factor
            accepted = True
        
        # reject step
        else:
            h *= max(minFactor, safety*error_norm**(1/p+1))
            accepted = False
        
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