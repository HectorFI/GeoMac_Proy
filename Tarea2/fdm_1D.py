import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


data = pd.read_csv('phys_dat.csv') # Lectura de datos: k, cp, rho
Dth_data = data['k'] / (data['rho']*data['cp']) # Cálculo del coeficiente Dth

print('\nDatos del problema \n{}'.format(data))
print('\n Dth : \n{}'.format(Dth_data))

def arithmeticMean(a, b):
    """
    Calcula la media aritmética entre a y b.
    
    Parameters
    ----------
    a, b: int
    Valores a interpolar.
    
    Returns
    -------
    La media aritmética.
    """
    return 0.5 * (a + b)

def harmonicMean(a, b):
    """
    Calcula la media harmónica entre a y b.
    
    Parameters
    ----------
    a, b: int
    Valores a interpolar.
    
    Returns
    -------
    La media harmónica.
    """    
    return 2 * a * b / (a + b)
    
def Laplaciano1D(N, d, f=None):
    """
    Calcula la matriz del Laplaciano usando diferencias finitas en 1D.
    
    Parameters
    ----------
    N: int
    Tamaño de la matriz
    
    d: float
    Valores del coeficiente Gamma de la ecuación.
    
    f: function
    Función para calcular el promedio entre dos valores.
    
    Returns
    -------
    A: ndarray
    La matriz del sistema.
    """
    if f == None:
        A = np.zeros((N,N))
        A[0, 0] = -2 * d[1]
        A[0, 1] = d[1]
        
        for i in range(1,N-1):
            A[i,i] = -2 * d[i+1]
            A[i,i+1] = d[i+1] #harmonicMean(d[i], d[i+1])
            A[i,i-1] = d[i] #harmonicMean(d[i], d[i+1])
            
        A[N-1,N-2] = d[N]
        A[N-1,N-1] = -2 * d[N]

    else:        
        A = np.zeros((N,N))
        A[0, 0] -= ( f(d[0], d[1]) + f(d[1], d[2]) )
        A[0, 1] = f(d[1], d[2])
        
        for i in range(1,N-1):
            A[i,i] -= ( f(d[i], d[i+1]) + f(d[i+1], d[i+2]) )
            A[i,i+1] = f(d[i+1], d[i+2])
            A[i,i-1] = f(d[i+1], d[i])

        A[N-1,N-2] = f(d[N-1], d[N])
        A[N-1,N-1] -= ( f(d[N-1], d[N]) + f(d[N], d[N+1]) )

    return A


def Laplaciano1D_NS(N, d, f=None):
    """
    Calcula la matriz del Laplaciano usando diferencias finitas en 1D.
    
    Parameters
    ----------
    N: int
    Tamaño de la matriz
    
    d: float
    Valores del coeficiente Gamma de la ecuación.
    
    f: function
    Función para calcular el promedio entre dos valores.
    
    Returns
    -------
    A: ndarray
    La matriz del sistema.
    """
    if f == None:
        A = np.zeros((N,N))
        A[0, 0] = ( 2 * d[1] + 1 )
        A[0, 1] = -d[1]

        for i in range(1,N-1):
            A[i,i] = ( 2 * d[i+1] + 1 )
            A[i,i+1] = -harmonicMean(d[i], d[i+1])
            A[i,i-1] = -harmonicMean(d[i], d[i+1])

        A[N-1,N-2] = -d[N]
        A[N-1,N-1] = ( 2 * d[N] + 1)

    else:     
        A = np.zeros((N,N))
        A[0, 0] = ( f(d[0], d[1]) + f(d[1], d[2]) + 1 )
        A[0, 1] = -f(d[1], d[2])    
        for i in range(1,N-1):
            A[i,i] = ( f(d[i], d[i+1]) + f(d[i+1], d[i+2]) + 1 )
            A[i,i+1] = -f(d[i+1], d[i+2])
            A[i,i-1] = -f(d[i+1], d[i])

        A[N-1,N-2] = -f(d[N-1], d[N])
        A[N-1,N-1] = ( f(d[N-1], d[N]) + f(d[N], d[N+1]) + 1)

    return A

def calcDth(Dth_data, z, N):
    """
    Calcula el Dth.
    
    Parameters
    ----------
    z, N: int
    
    """
    Dth = np.zeros((N+2))
    for k in range(0, N+2):
        if (z[k] <= 50.0):
            Dth[k] = Dth_data[0]
        elif ((z[k] > 50.0) and (z[k] <= 250.0)):
            Dth[k] = Dth_data[1]
        elif ((z[k] > 250.0) and (z[k] <= 400.0)):
            Dth[k] = Dth_data[2]
        elif ((z[k] > 400.0) and (z[k] <= 600.0)):
            Dth[k] = Dth_data[3]
        elif ((z[k] > 600.0) and (z[k] <= 800.0)):
            Dth[k] = Dth_data[4]
        elif ((z[k] > 800.0) and (z[k] <= 1000.0)):
            Dth[k] = Dth_data[5]
        elif ((z[k] > 1000.0) and (z[k] <= 1500.0)):
            Dth[k] = Dth_data[6]
        elif ((z[k] > 1500.0) and (z[k] <= 1900.0)):
            Dth[k] = Dth_data[7]
        else:
            Dth[k] = Dth_data[8]
            
            
    return Dth

def interpTemp(z_dat, T_dat):
    """
    Calcula la interpolacion de los datos de funcion.
    
    Parameters
    ----------
    z_dat, T_dat: int
    
    """
    
    z_dat = [0, 100, 200, 400, 710, 803, 1100, 1200, 1400, 1500, 1600, 1700, 1800, 2000, 2500, 3000, 3500, 4000]
    
    T_dat = [15, 113, 145, 178, 155, 201, 215, 282, 223, 226, 252, 284, 310, 350, 450, 550, 650, 750]
    tck_1 = interpolate.splrep(z_dat, T_dat, s = 0)
 
    return tck_1