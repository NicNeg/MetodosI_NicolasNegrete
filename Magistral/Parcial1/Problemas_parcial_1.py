# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:28:55 2023

@author: elect
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from pathlib import Path
import wget





#Derivación_problema 8:
    
x_d_8 = np.linspace(0.1,1.1,101)
h_d_8 = 0.01

def raiz_tangente(x):

    return np.sqrt(np.tan(x))

def d_progresiva_orden2(x,h):
    
    d = (-3*raiz_tangente(x) + 4*raiz_tangente(x + h) - raiz_tangente(x + (2*h)))/(2*h)
    
    return d

def d_central(x,h):
    
    d = (raiz_tangente(x + h) - raiz_tangente(x - h))/(2*h)
    
    return d

def d_exacta(x):
    
    return 1/((2*np.sqrt(np.tan(x)))*(np.cos(x)**2))

de = d_exacta(x_d_8)
dp = d_progresiva_orden2(x_d_8, h_d_8)
dc = d_central(x_d_8, h_d_8)

def graficar_derivadas(x):
    
    plt.scatter(x,dp, s= 50, color = 'orange', label = 'Derivada Progresiva de orden O(h^2)')
    plt.scatter(x,dc, s= 50, label = 'Derivada Central')
    plt.scatter(x,de, s= 10, color = 'r', label = 'Derivada Exacta')
    plt.legend()
    plt.show()
    
    '''Los puntos de las derivadas progresiva y central se pusieron con un tamaño mayor para poder notar mejor
    a la derivada exacta, el tamaño solo es algo visual y no tiene correlación con los valores'''

#graficar_derivadas(x_d_8)

def graficar_errores(x):
    
    error_dp = np.abs(de-dp)
    error_dc = np.abs(de-dc)
    
    plt.scatter(x,error_dp, color = 'orange', label = 'Error de la derivada progresiva de orden O(h^2)')
    plt.scatter(x,error_dc, label = 'Error de la derivada central')
    plt.legend()
    plt.show()
    
    '''Como se puede apreciar, la derivada central sigue teniendo menor error que la progresiva aún cuando
    esta posee un error menor a su forma original. Con esto en mente, tiene sentido usar la central en lugar
    de la progresiva o regresiva debido a su mayor eficiencia en dar resultados más cercanos a los verdaderos.'''

#graficar_errores(x_d_8)





#Raíces de polinomios_problema 3:

x_p_3 = np.linspace(-2,1.2,100)
h_p_3 = x_p_3[1] - x_p_3[0]

def funcion(x):
    
    f = 3*(x**5) + 5*(x**4) - (x**3)
    
    return f

def derivada_exacta(x):
    
    d = 15*(x**4) + 20*(x**3) - 3*(x**2)
    
    return d

def derivada_central(x,h):
    
    dc = (funcion(x + h) - funcion(x - h))/(2*h)
    
    return dc
    
f = funcion(x_p_3)
d = derivada_exacta(x_p_3)
dc = derivada_central(x_p_3, h_p_3)

def metodo_newton_raphson(xn,h):
    
    error = 1.0
    precision = 1e-8
    it = 0
    itmax= 100
    
    while error > precision and it < itmax:
        
        try:
        
            xn1 = xn - ((funcion(xn))/(derivada_central(xn, h)))
            error = np.abs((xn1 - xn)/xn1)
        
        except ZeroDivisionError:
            
            print('División por 0')
            
        xn = xn1
        it += 1
        
    return xn

#raiz = metodo_newton_raphson(-1.2888888888888888, h_p_3)

def todas_las_raices(x,h):
    
    raices = np.array([])
    tolerancia = 7
    
    for i in x:
        
        raiz = metodo_newton_raphson(i,h)
        craiz = np.round(raiz,tolerancia)
        
        if craiz not in raices:
            
            raices = np.append(raices,craiz)
            
    raices.sort()
    
    return raices

#raices = todas_las_raices(x_p_3, h_p_3)

'''

def graficar(x):
    
    plt.plot(x,f)
    plt.axhline(0, color = 'r')
    plt.show()
    
graficar(x_p_3)
'''





#Interpolación de Lagrange_problema 4:

x_l_4 = np.linspace(0,7,1000)
g = 9.8
url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Parabolico.csv'
n_archivo = 'Parabolico.csv'

def conseguir_archivo(url):
    
    ruta = Path.cwd() / 'Parabolico.csv'
    
    if not path.exists(ruta):
        
        archivo = wget.download(url,n_archivo)
        
    else:
        
        l_ruta = str(ruta)
        ll_ruta = l_ruta.split('\\')
        archivo = ll_ruta[len(ll_ruta)-1]
    
    return archivo

archivo = conseguir_archivo(url)

def datos(archivo):
    
    datos = pd.read_csv(archivo, sep = ',')
    
    return datos

datos = datos(archivo)

def Interpolacion_Lagrange(x,datos):
    
    array_x = np.array([])
    array_y = np.array([])
    P = 0
    
    for i in range(datos.shape[0]):
        
        x_datos = datos.iloc[i,0]
        y_datos = datos.iloc[i,1]
        array_x = np.append(array_x,x_datos)
        array_y = np.append(array_y,y_datos)
        
    for i in range(array_x.shape[0]):
        
        L = 1
        
        for j in range(array_x.shape[0]):
        
            if i != j:
                
                L *= (x - array_x[j])/(array_x[i] - array_x[j])
                
        P += L*array_y[i]
        
    plt.scatter(array_x,array_y, color ='g')
    plt.plot(x,P)
    plt.plot(x,np.interp(x, array_x, array_y),'--',color = 'orange')
    plt.axhline(0,color= 'r')
    plt.show()
    
    return P

#P = Interpolacion_Lagrange(x_l_4,datos)

def hallar_velocidad_inicial_angulo(x,P):
    
    y = P.max()
    valor = np.where(P == y)[0]
    
    Vy = np.sqrt(2*g*y)    
    t = Vy/g
    
    x = x[valor]
    x_v = x.max()
    
    Vx = x_v/t
    V = np.sqrt(Vx**2 + Vy**2)
    Theta = np.arctan(Vy/Vx)*(360/(2*np.pi))
    
    tupla = (V,Theta)
    
    return tupla

#tupla_v_theta = hallar_velocidad_inicial_angulo(x_l_4,P)
