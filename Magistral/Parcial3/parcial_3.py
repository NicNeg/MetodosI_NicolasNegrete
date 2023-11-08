# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:04:01 2023

@author: elect
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

#Punto 23

#A

A = sym.Matrix([[0.2,0.1,1,1,0], [0.1,4,-1,1,-1], [1,-1,60,0,-2], [1,1,0,8,4], [0,-1,-2,4,700]])
b = sym.Matrix([[1,2,3,4,5]])

x0 = sym.Symbol('x0',real=True)
x1 = sym.Symbol('x1',real=True)
x2 = sym.Symbol('x2',real=True)
x3 = sym.Symbol('x3',real=True)
x4 = sym.Symbol('x4',real=True)

x = sym.Matrix([x0, x1, x2, x3, x4])

#B_&_C

def descenso_conjugado(A,x,b,itmax=5):
    
    tolerancia = 0.01
    it = 0
    r0 = A.dot(x[0,0]) - b
    p0 = -r0
    norma_r = np.linalg.norm(r0)
    a_k = 0
    x_k_vector = []
    i = 0
    
    while norma_r > tolerancia and it < itmax:
        
        if a_k == 0:
            
            a_k = -(((r0.transpose())*p0)/((p0.transpose())*A*p0))
            x_k = x[i,0] + a_k*p0
            x_k_vector.append(x_k)
        
        else: 
            
            if len(x_k_vector) == 1:
            
                r_k = A*x_k - b
                B_k = ((r_k.transpose())*A*p0)/((p0.transpose())*A*p0)
                p_k = -r_k + B_k*p0
                i += 1
                a_k = -(((r_k.transpose())*p_k)/((p_k.transpose())*A*p_k))
                x_k = x[i,0] + a_k*p_k
                
            else:
                
                r_k = A*x_k - b
                B_k = ((r_k.transpose())*A*p_k)/((p_k.transpose())*A*p_k)
                p_k = -r_k + B_k*p_k
                i += 1
                a_k = -(((r_k.transpose())*p_k)/((p_k.transpose())*A*p_k))
                x_k = x[i,0] + a_k*p_k
        
        it +=1
        
    return x_k_vector
    
#descenso_conjugado(A, x, b)