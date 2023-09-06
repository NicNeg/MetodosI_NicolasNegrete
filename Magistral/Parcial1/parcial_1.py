# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:54:38 2023

@author: elect
"""

import numpy as np
import pandas as pd
from os import path
from pathlib import Path
import wget
import sympy as sym
import matplotlib.pyplot as plt

#xg se usa para graficar el intervalo aprox de la raiz

xg = np.linspace(0.5670,0.5672,1000)

def funcion(x):
    
    f = (np.e**(-x))-x
    
    return f 

f = funcion(xg)

#Punto B

x0 = 0.0
x1 = 1.0

def teorema_boltzano():
    
    f0 = funcion(x0)
    f1 = funcion(x1)
    
    v = f0*f1
    
    if v < 0:
        
        print('Se cumple y hay una raiz entre los puntos')
        
    if v > 0:
        
        print('No se cumple')
        
teorema_boltzano()

#Como se cumple el teorema, podemos avanzar

#Punto C

x2 = (x0+x1)/2

#Punto D

X = np.array([x0,x2,x1])
Y = np.array([funcion(x0),funcion(x2),funcion(x1)])
h = X[1] - X[0]

def crear_matriz_diferencias():
    
    Diff = np.zeros((X.shape[0],Y.shape[0]))
    Diff[:,0] = Y
    
    for i in range(1,len(X)):
        for j in range(i,len(X)):
            Diff[j,i] = Diff[j,i-1] - Diff[j-1,i-1] 
            
    return Diff

Diff = crear_matriz_diferencias()

def muller():
    
    a = Diff[2,2]
    
    b = Diff[1,1] - (x0 + x1)*Diff[2,2]
    
    c = funcion(x0) - x0*Diff[1,1] + x0*x1*Diff[2,2]
    
    if b < 0:
        
        x3 = (-2*c)/(b-(np.sqrt((b**2)-4*a*c)))
        
    if b > 0:
        
        x3 = (-2*c)/(b+(np.sqrt((b**2)-4*a*c)))
        
    if b == 0: 
        
        x3 = (-2*c)/(b+(np.sqrt((b**2)-4*a*c)))
    
    return x3

x3 = muller()

#Punto E

def muller_iterable():
    
    x3 = 0
    xp2 = (x0+x1)/2
    itmax = 100
    it = 1
    precision = 1e-10
    ep = np.abs(funcion(xp2))
    
    while ep > precision and it < itmax:
        
        X2 = np.array([x0,x1,xp2])
        Y2 = np.array([funcion(x0),funcion(x1),funcion(xp2)])
        Diff2 = np.zeros((X2.shape[0],Y2.shape[0]))
        Diff2[:,0] = Y2
    
    
        for j in range(1,len(X2)):
            w = 0
            for k in range(j,len(X2)):
                Diff2[k,j] = (Diff2[k,j-1] - Diff2[k-1,j-1])/(X2[k] - X2[w])
                w += 1
                
                
        a = Diff2[2,2]
        
        b = Diff2[1,1] - (x0 + x1)*Diff2[2,2]
        
        c = funcion(x0) - x0*Diff2[1,1] + x0*x1*Diff2[2,2]
        
        if b < 0:
            
            x3 = (-2*c)/(b-(np.sqrt((b**2)-4*a*c)))
            
        if b > 0:
            
            x3 = (-2*c)/(b+(np.sqrt((b**2)-4*a*c)))
            
        if b == 0: 
            
            x3 = (-2*c)/(b+(np.sqrt((b**2)-4*a*c)))
                
        xp2 = x3
        ep = np.abs(funcion(xp2))
        it += 1
        
    raiz = np.round(xp2,10)
        
    return raiz

raiz = muller_iterable()

#Punto A y F
            
#La raiz segun el algoritmo es aprox 00.5671432904, como se puede apreciar en la grafica

def graficar(x):
    
    plt.plot(x,f)
    plt.axhline(0, color = 'r')
    plt.scatter(raiz,0,color = 'g')
    plt.show()
    
graficar(xg)