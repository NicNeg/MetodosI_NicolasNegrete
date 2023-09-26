# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:24:21 2023

@author: elect
"""

import numpy as np
import matplotlib.pyplot as plt

#Punto 1.1

t = np.linspace(0,1.85675,100) # Para dos bucles
h = t[1] - t[0]
B_0 = 0.05
r = 12.5
w = 3.5
f = 7
R = 1.75

def funcion_B(t):
    
    B = np.pi*(r**2)*B_0*np.cos(w*t)*np.cos(2*np.pi*f*t)
    
    return B

def derivada_central_B(t):
    
    dc = (funcion_B(t + h) - funcion_B(t - h))/(2*h)
    
    return dc

B = funcion_B(t)
dc = derivada_central_B(t)

def corriente(t):
    
    I = (-1/R)*(derivada_central_B(t))
    
    return I

I = corriente(t)
    
#Punto 1.2

def bif_todas_las_raices(x):
    
    lista_rango = [(1,5),(6,11)]
    lista_raices = np.array([0])
    tolerancia = 7
    
    for i in lista_rango:
            
        a = i[0]
        b = i[1]
        raiz_vd = corriente(x[a])
            
        for j in x[a:b]:
            
            for k in reversed(x[a:b]):
                
                c = (j+k)/2
                raiz_f = corriente(c)
                    
                resta = np.abs(raiz_vd) - np.abs(raiz_f)
                    
                if resta > 0:
                        
                    raiz_vd = raiz_f
                    c_vd = c
                        
        cc = np.round(c_vd,tolerancia)
        lista_raices = np.append(lista_raices,cc)
        lista_raices.sort()
    
    return lista_raices

raices = bif_todas_las_raices(t)

Y = [0,0,0]

def graficar():
    
    plt.plot(t,I)
    plt.axhline(0, color = 'r')
    plt.axvline(0,color = 'r')
    plt.scatter(raices,Y, color = 'g')
    plt.show()
    
graficar()