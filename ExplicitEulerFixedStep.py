def ExplicitEulerFixedStep(fun, t0, tN, N, x0, *args):
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