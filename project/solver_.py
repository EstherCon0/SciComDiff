import numpy as np

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
    def __init__(self, gamma:float=(1-(1/2**0.5)), rtol=1e-6, atol=1e-9, h_init=1e-2, h_min=1e-6, h_max=1.0, max_iter=10, tol=1e-8):
        self.gamma = gamma
        self.rtol = rtol
        self.atol = atol
        self.h_init = h_init
        self.h_min = h_min
        self.h_max = h_max
        self.max_iter = max_iter
        self.tol = tol

        # Coefficients for ESDIRK23 (L-stable); common choice gamma = 1 - 1/sqrt(2) = 0.2928932188134524
        self.c = [0, self.gamma, 1.0]
        self.A = np.array([
            [self.gamma, 0, 0],
            [1 - self.gamma, self.gamma, 0],
            [0.25, 0.25, self.gamma]
        ])
        self.b = np.array([0.25, 0.25, self.gamma])
        self.b_hat = np.array([0.5, 0.5, 0.0])  # Lower-order for error estimate
        self.e = self.b - self.b_hat

    def step(self, f, t, y, jac=None, params=None, h=None):
        if h is None:
            h = self.h_init
        s = len(self.c)
        k = [np.zeros_like(y) for _ in range(s)]

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
                    dki = np.linalg.solve(I - h * self.A[i][i] * J, res)
                    ki_new = ki - dki
                else:
                    ki_new = g  # basic fixed-point update

                if np.linalg.norm(ki_new - ki) < self.tol:
                    break
                ki = ki_new

            k[i] = ki

        y_new = y + h * sum(self.b[i] * k[i] for i in range(s))
        y_hat = y + h * sum(self.b_hat[i] * k[i] for i in range(s))
        err = np.linalg.norm((y_new - y_hat) / (self.atol + self.rtol * np.abs(y_new)), ord=np.inf)
        return y_new, err

    def integrate(self, f, t0, y0, tf, jac=None, params=None):
        t_vals = [t0]
        y_vals = [y0]
        t, y, iter_ = t0, np.array(y0), 1
        h = self.h_init

        while t < tf:
            # Adjust step to not overshoot
            h = min(h, tf - t)

            y_new, err = self.step(f, t, y, jac=jac, params=params, h=h)

            if err <= 1:
                # Accept step
                t += h
                y = y_new
                t_vals.append(t)
                y_vals.append(y.copy())

                # Adapt step size
                h = min(self.h_max, h * min(5.0, 0.9 * (1 / err)**0.2)) if err > 0 else self.h_max
            else:
                # Reject step
                h = max(self.h_min, h * max(0.1, 0.9 * (1 / err)**0.25))

            # Ensure progress to avoid infinite loop
            if h < 1e-12:
                raise RuntimeError("Step size became too small — integration may be stuck.")
            iter_+=1
            if iter_%500==0: print(f"Iteration {iter_}, err: {err}, h: {h}")

        return np.array(t_vals), np.array(y_vals)

