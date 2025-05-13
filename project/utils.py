import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def StdMultivariateWienerProcess(T, N, nW, Ns, seed=None):
    """
    Generates Ns realizations of a standard Wiener process.
    
    Parameters:
    T    : Final time
    N    : Number of intervals
    nW   : Dimension of W(k)
    Ns   : Number of realizations
    seed : To set the random number generator (optional)
    
    Returns:
    W    : Standard Wiener process in [0,T]
    Tw   : Time points
    dW   : White noise used to generate the Wiener process
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(nW, N, Ns)
    W = np.concatenate((np.zeros((nW, 1, Ns)), np.cumsum(dW, axis=1)), axis=1)
    Tw = np.linspace(0, T, N + 1)
    
    return W, Tw, dW

def RK_stability(A:np.ndarray, b:np.ndarray, err_vec:np.ndarray, grid_size:int=5, pixels:int=500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the stability function of a generic RK method with Butcher Tableau.
      Based on the 'test-equation': x_dot(t) = \lambda * x(t).

    Args:
        A (np.ndarray): RK method Butcher-Coefficient Matrix A
        b (np.ndarray): RK method Butcher-Coefficient vector b
        err_vec (np.ndarray): for integrated and embedded RK methods, Butcher-Error vector e, sometimes called d
        grid_size (int, optional): Defaults to 5.
        pixels (int, optional): Defaults to 500.

    Returns:
        Tuple[
            absR (np.ndarray): |R(z)| absolute value of the complex stability function, 
            absEhat (np.ndarray): absolute value of the error stability, 
            absEhatmE (np.ndarray): |E_hat(z) - (R(z)-exp(z))|,
            absE (np.ndarray): |R(z) - exp(z)|, 
            absF (np.ndarray): |exp(z)| - reference for L-stability
        ]
    """
    x, y = np.linspace(-grid_size, grid_size, pixels), np.linspace(-grid_size, grid_size, pixels)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    absR, absEhat, absEhatmE, absE, absF = [], [], [], [], []
    for z in Z.flatten():
        A_inv_ = np.linalg.solve((np.eye(len(A))-z*A), np.ones(len(b)))
        R = 1+np.inner(z,b.T)@A_inv_
        Ehat = np.inner(z,err_vec.T)@A_inv_
        E = R-np.exp(z)
        absR.append(np.abs(R))
        absEhat.append(np.abs(Ehat))
        absEhatmE.append(np.abs(Ehat-E))
        absE.append(np.abs(E))
        absF.append(np.abs(np.exp(z)))
    absR = np.array(absR).reshape(pixels, pixels)
    absEhat = np.array(absEhat).reshape(pixels, pixels)
    absEhatmE = np.array(absEhatmE).reshape(pixels, pixels)
    absE = np.array(absE).reshape(pixels, pixels)
    absF = np.array(absF).reshape(pixels, pixels)
    
    return absR, absEhat, absEhatmE, absE, absF

def plot_RK_stability(A:np.ndarray, b:np.ndarray, err_vec:np.ndarray, method_name:str=None, grid_size:int=5, pixels:int=500):
    """Calculates AND plots the stability metrics of a generic RK method with Butcher Tableau.
      Based on the 'test-equation': x_dot(t) = \lambda * x(t).

    Args:
        A (np.ndarray): RK method Butcher-Coefficient Matrix A
        b (np.ndarray): RK method Butcher-Coefficient vector b
        err_vec (np.ndarray): for integrated and embedded RK methods, Butcher-Error vector e, sometimes called d
        method_name (str): Name of the method for plot title. Defaults to None.
        grid_size (int, optional): Defaults to 5.
        pixels (int, optional): Defaults to 500.

    Returns: None, generates plot.
    """
    
    absR, absEhat, absEhatmE, absE, absF = RK_stability(A, b, err_vec, grid_size, pixels)

    fig = plt.figure(figsize=(10,4), dpi=400)
    if method_name!=None:
        fig.suptitle(f"Stability for {method_name}")
    ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(shape=(2, 4), loc=(0, 2))
    ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 3))
    ax4 = plt.subplot2grid((2, 4), (1, 2))
    ax5 = plt.subplot2grid((2, 4), (1, 3))

    ax1.set_title('$|R(z)|$')
    ax1.imshow(absR.clip(max=1))
    ax1.axvline(pixels//2, color='r')
    ax1.axhline(pixels//2, color='r')
    ax1.set_xticks(ticks=np.arange(0, pixels, 50), labels=np.arange(-grid_size, grid_size,1))
    ax1.set_yticks(ticks=np.arange(0, pixels, 50), labels=np.arange(-grid_size, grid_size,1))
    ax1.set_xlabel("$Re(z)$")
    ax1.set_ylabel("$Im(z)$")

    ax2.set_title('$|\hat{E}(z)|$')
    ax2.set_xticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax2.set_yticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax2.axvline(pixels//2, color='r')
    ax2.axhline(pixels//2, color='r')
    ax2.imshow(absEhat.clip(max=1))

    ax3.set_title('$|exp(z)|$')
    ax3.set_xticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax3.set_yticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax3.axvline(pixels//2, color='r')
    ax3.axhline(pixels//2, color='r')
    ax3.imshow(absF.clip(max=1))
    
    ax4.set_title('$|E(z)|$')
    ax4.set_xticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax4.set_yticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax4.axvline(pixels//2, color='r')
    ax4.axhline(pixels//2, color='r')
    ax4.imshow(absE.clip(max=1))

    ax5.set_title('$|E(z) - \hat{E}(z)|$')
    ax5.set_xticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax5.set_yticks(ticks=np.arange(0, pixels, 50), labels=[])
    ax5.axvline(pixels//2, color='r')
    ax5.axhline(pixels//2, color='r')
    ax5.imshow(absEhatmE.clip(max=1))

    fig.subplots_adjust(hspace=0.25)
    fig.colorbar(plt.imshow(absE.clip(max=1)), ax=ax1)