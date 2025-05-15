import numpy as np
import json, os
from datetime import datetime

class DOPRI54:
    def __init__(self, rtol=1e-6, atol=1e-9, h_init=0.01, h_min=1e-6, h_max=1.0):
        self.rtol = rtol
        self.atol = atol
        self.h_init = h_init
        self.h_min = h_min
        self.h_max = h_max

        # Butcher tableau for Dormand-Prince 5(4)
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
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

    def step(self, f, t, y, params=None, h=None):
        if h is None:
            h = self.h_init
        k = []
        for i in range(7):
            yi = y.copy()
            for j in range(i):
                yi += h * self.A[i][j] * k[j]
            k.append(f(t + self.c[i] * h, yi, *params) if params else f(t + self.c[i] * h, yi))

        y_next = y + h * np.dot(self.b, k)
        y_err = h * np.dot(self.b - self.b_hat, k)
        err_norm = np.linalg.norm(y_err / (self.atol + self.rtol * np.abs(y_next)), ord=np.inf)
        return y_next, err_norm, k[-1]

    def integrate(self, f, t0, y0, tf, params=None):
        t_vals = [t0]
        y_vals = [y0]
        t = t0
        y = y0
        h = self.h_init

        while t < tf:
            if t + h > tf:
                h = tf - t
            y_new, err_norm, _ = self.step(f, t, y, params, h)

            if err_norm <= 1:
                t += h
                y = y_new
                t_vals.append(t)
                y_vals.append(y)
                h = min(self.h_max, h * min(5, 0.9 * (1 / err_norm)**0.2))
            else:
                h = max(self.h_min, h * max(0.1, 0.9 * (1 / err_norm)**0.25))

        return np.array(t_vals), np.array(y_vals)


class ESDIRK23:
    def __init__(self, gamma:float=(2-2**0.5)/2, rtol=1e-6, atol=1e-9, h_init=1e-2, h_min=1e-6, h_max=1.0, max_iter=10, tol=1e-8, eps:float=0.8, norm:str|int='inf'):
        self.gamma = gamma # commonly (1-(1/2**0.5)) \approx (2-2**0.5)/2
        self.rtol = rtol
        self.atol = atol
        self.h_init = h_init
        self.h_min = h_min
        self.h_max = h_max
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.norm = norm
        self.info = {'JacDir_ki_diff':[], 'ESDIRK_k': [], 't':[], 'step_evals':0, 'accepted_iter':0, 'rejected_iter':0, 'h_step_width':[h_init, h_init], 'rel_err_norm':[1e6, 1e6]}

        # ESDIRK23 constant properties
        self.p = 2      # integrated order
        self.phat = 3   # embedded order
        self.s = 3      # stages

        # Coefficients for ESDIRK23 (L-stable); common choice gamma = 1 - 1/sqrt(2) = 0.2928932188134524
        self.c = [0, 2*self.gamma, 1.0]
        self.A = np.array([
            [0, 0, 0],
            [self.gamma, self.gamma, 0],
            [(1-self.gamma)/2, (1-self.gamma)/2, self.gamma]
        ])
        self.b = np.array([(1-self.gamma)/2, (1-self.gamma)/2, self.gamma])
        self.b_hat = np.array([(6*self.gamma-1)/(12*self.gamma), 1/(12*self.gamma*(1-2*self.gamma)), (1-3*self.gamma)/(3*(1-2*self.gamma))])  # higher-order embedded for error estimate
        self.e = self.b - self.b_hat

    def step(self, f, t, y, jac=None, params=None, h=None):
        if h is None:
            h = self.h_init
        s = len(self.c)
        k = [np.zeros_like(y) for _ in range(s)]
        self.info['step_evals'] += 1

        # iter through stages
        for i in range(s):
            ti = t + self.c[i] * h
            yi = y.copy()
            for j in range(i):
                yi += h * self.A[i][j] * k[j]

            # Solve implicit equation: ki = f(ti, yi + h * a_ii * ki)
            ki = f(ti, yi, *params) if params else f(ti, yi)
            for _ in range(self.max_iter):
                g = f(ti, yi + h * self.A[i][i] * ki, *params) if params else f(ti, yi + h * self.A[i][i] * ki)
                if jac:
                    J = jac(ti, yi + h * self.A[i][i] * ki, *params) if params else jac(ti, yi + h * self.A[i][i] * ki)
                    I = np.eye(len(y))
                    res = ki - g
                    #print("res:", res, "J", J, "ki", ki, "g", g)
                    #print('solve matrix:', I - h * self.A[i][i] * J)
                    dki = np.linalg.solve(I - h * self.A[i][i] * J, res)
                    ki_new = ki - dki
                else:
                    ki_new = g  # basic fixed-point update

                ki_diff = np.linalg.norm(ki_new - ki)
                self.info['JacDir_ki_diff'].append(ki_diff)
                if  ki_diff < self.tol:
                    break
                ki = ki_new
            
            k[i] = ki
            self.info['ESDIRK_k'].append(ki.tolist())

        y_new = y + h * sum(self.b[i] * k[i] for i in range(s))
        y_hat = y + h * sum(self.b_hat[i] * k[i] for i in range(s))

        if self.norm=='inf':
            err = np.linalg.norm((y_new - y_hat) / (self.atol + self.rtol * np.abs(y_new)), ord=np.inf)
        elif type(self.norm)==int:
            err = np.linalg.norm((y_new - y_hat) / (self.atol + self.rtol * np.abs(y_new)), ord=self.norm)

        return y_new, err

    def integrate_adaptive(self, f, t0, y0, tf, jac=None, params=None, verbose=False, save_traces:bool=False, save_trace_path:str="./project/traces", controller:str='aggressive'):
        allowed_controller = ['aggressive', 'moderate', 'pid', 'asymptotic', 'predictive']
        if controller not in allowed_controller: raise ValueError("The controller must be either: 'aggressive', 'moderate', 'pi', 'asymptotic'")

        self.info['t'].append(t0)
        save_path = os.path.join(save_trace_path, 'ESDIRK23_adaptive_'+datetime.now().strftime('%d_%m_%Y__%H_%M_%S')+'.json')
        
        y_vals = [np.array(y0)]
        t, y, iter_ = t0, np.array(y0), 1
        h = self.h_init

        while t < tf:
            self.info['h_step_width'].append(h)
            h = min(h, tf - t)

            # integration step safeguard
            try:
                y_new, err_norm_tol = self.step(f, t, y, jac=jac, params=params, h=h)
                self.info['rel_err_norm'].append(err_norm_tol)
            except Exception as e:
                raise RuntimeError(f"ESDIRK23 integration step failed at t={t} with h={h}: {e}")

            if controller=='aggressive':
                # accept step
                if err_norm_tol <= 1:
                    self.info['accepted_iter'] += 1
                    t += h
                    y = y_new
                    self.info['t'].append(t)
                    y_vals.append(y.copy())

                    # Adapt step size aggressively if error is small
                    if err_norm_tol == 0:
                        h_new = h * 2.0
                    else:
                        h_new = h * min(5.0, max(1.5, 0.9 * (1 / err_norm_tol)**0.3))

                    h = min(self.h_max, h_new)
                else:
                    # Reject step and shrink
                    self.info['rejected_iter'] += 1
                    h = max(self.h_min, h * max(0.1, 0.9 * (1 / err_norm_tol)**0.25))
                
            elif controller=='moderate':
                if err_norm_tol <= 1:
                    t += h
                    y = y_new
                    self.info['t'].append(t)
                    y_vals.append(y)
                    h = min(self.h_max, h * min(5, 0.9 * (1 / err_norm_tol)**0.2)) if err_norm_tol > 0 else self.h_max
                else:
                    h = max(self.h_min, h * max(0.1, 0.9 * (1 / err_norm_tol)**0.25))

            elif controller=='asymptotic':
                t += h
                y = y_new
                self.info['t'].append(t)
                y_vals.append(y)
                h *= (self.eps/err_norm_tol)**(1/(self.phat))

            elif controller=='pid':
                t += h
                y = y_new
                self.info['t'].append(t)
                y_vals.append(y)
                pass

            elif controller=='predictive':
                t += h
                y = y_new
                self.info['t'].append(t)
                y_vals.append(y)

                k2, k1 = 1, 1 # after Gustaffson (1992); Sensitivity Paper Eq.40
                past_h_ratio = h/self.info['h_step_width'][-2]
                past_r_ratio = self.info['rel_err_norm'][-2]/err_norm_tol

                h *= ((past_h_ratio)*(self.eps/err_norm_tol)**(k2/self.phat)) * (past_r_ratio**(k1/self.phat))

            # Guard against vanishing h
            if h < 1e-12:
                raise RuntimeError("Step size too small — integration may be stuck.")

            iter_+=1
            if iter_%500==0 and verbose: print(f"Iteration {iter_}, err: {err_norm_tol}, h: {h}, t: {t}")
            if iter_%1000==0 and save_traces:
                with open(save_path, 'w') as fp:
                    json.dump(self.info, fp)

        return np.array(y_vals), self.info
    
    def integrate_fixed(self, f, t0, y0, tf, h, jac=None, params=None, verbose:bool=False):
        self.info['t'].append(t0)
        y_vals = [np.array(y0)]
        t, y, iter_ = t0, np.array(y0), 1

        while t < tf:
            h = min(h, tf - t)
            self.info['h_step_width'].append(h)
            y, err_norm  = self.step(f, t, y, jac=jac, params=params, h=h)
            t += h
            self.info['t'].append(t)
            self.info['rel_err_norm'].append(err_norm)
            y_vals.append(y.copy())
            iter_+=1
            if verbose: print(f'iteration: {iter_}')

        return np.array(y_vals), self.info
