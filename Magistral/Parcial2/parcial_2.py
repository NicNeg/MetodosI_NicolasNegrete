# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:56:48 2023

@author: elect
"""

import sympy as sym
import numpy as np

#Elegi del punto 23

h = 6.626*(10**-34)
k = 1.3806*(10**-23)
c = 3*(10**8)
T = 5772
L_0 = (100/(10**9))
L_1 = (400/(10**9))
v_0 = c/L_0
v_1 = c/L_1

limite_1 = (h*v_1)/(k*T)
limite_0 = (h*v_0)/(k*T)

def funcion_base(x):
    
    f = (x**3)/((np.exp(x))-1)
    
    return f

#Punto a
    
def funcion_numerador(z):
    
    f = funcion_base(((limite_0 - limite_1)/2)*z + ((limite_0 + limite_1)/2))
    
    return f

def Legendre(n):
    
    Leg = np.polynomial.legendre.leggauss(n)
    
    return Leg

def Integral_numerador(n):
    
    tupla = Legendre(n)
    raices = tupla[0]
    pesos = tupla[1]
    valor = 0
    
    for i in range(n):
    
         valor += pesos[i]*funcion_numerador(raices[i])
         
    I = ((limite_1 - limite_0)/2)*valor
    
    return I


#I_numerador = Integral_numerador(20)
#print(I_numerador)

#Punto b

def funcion_denominador(x):
    
    f = (np.exp(x))*(funcion_base(x))
    
    return f

def Laguerre(n):
    
    Lag = np.polynomial.laguerre.laggauss(n)
    
    return Lag
        
def Integral_denominador(n):
    
    tupla = Laguerre(n)
    raices = tupla[0]
    pesos = tupla[1]
    valor = 0
    
    for i in range(n):
    
         valor += pesos[i]*funcion_denominador(raices[i])
         
    I = valor
    
    return I

#I_denominador = Integral_denominador(20)        
#print(I_denominador)

#Punto c

#print(limite_1)
#print(limite_0)

#Toca invertir los limites debido a que el limite_1 es menor al limite_0, una integral va del menor numero al mayor numero.

#Punto d

def fraccion_de_rayos(n):
    
    f = Integral_numerador(n)/Integral_denominador(n)
    
    return f

#fraccion_uv_porcentaje = np.abs(fraccion_de_rayos(20))*100
#print(fraccion_uv_porcentaje)

#Punto e

'''
La diferencia se debe al hecho de que nuestras integrales no son las debidas. Nosotros usamos integrales 
simplificadas como no conocemos ni la fisica ni los metodos para hallar las integrales originales.
Por lo que en la simplificación habra cierto grado de suposición que impedira hallar la fracción de rayos correcta.
'''



#Punto 25

x1 = sym.Symbol('x',real=True)

#punto_a

def GetLaguerre(k,x):
    
    if k == 0:
        
        poly = sym.Number(1)
        
    elif k == 1:
        
        poly = 1 - x
        
    else:
        
        poly = ((((2*k)-1-x)*GetLaguerre(k-1, x))-((k-1)*GetLaguerre(k-2, x)))/k
     
    return sym.expand(poly,x)

#poli_2 = GetLaguerre(2, x1)
#print(poli_2)

#punto_b

def GetLaguerre_Rodriguez(n,x):
    
    f = sym.exp(-x)*(x**n)
    df = sym.diff(f, x)
    df2 = sym.diff(df, x)
    
    Lag = ((sym.exp(x))/n)*df2
    
    return sym.expand(Lag,x)

def GetNewton(f,df,xn,itmax=10000,precision=1e-14):
    
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(xn)
            
            error = np.abs(f(xn)/df(xn))
            
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        return False
    else:
        return xn
    
def GetRoots(f,df,x,tolerancia = 10):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)

        if  type(root)!=bool:
            croot = np.round( root, tolerancia )
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

def GetDLaguerre(n,x):
    
    Pn = GetLaguerre(n,x)
    return sym.diff(Pn,x,1)

def GetAllRootsGLag(n):
    
    c_s = n+(n-1)*np.sqrt(n)
    
    xn = np.linspace(0,c_s,1000)
    
    Laguerre = []
    DLaguerre = []
    
    for i in range(n+1):
        Laguerre.append(GetLaguerre_Rodriguez(i,x1))
        DLaguerre.append(GetDLaguerre(i,x1))
    
    poly = sym.lambdify([x1],Laguerre[n],'numpy')
    Dpoly = sym.lambdify([x1],DLaguerre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots

#raices_2 = GetAllRootsGLag(2)
#print(raices_2)

#punto_c

def bases_cardinales(n,x):
    
    raices = GetAllRootsGLag(n)
    raiz_1 = raices[0]
    raiz_2 = raices[1]
    
    b1 = (x - raiz_1)/(raiz_1 - raiz_2)
    b2 = (x - raiz_2)/(raiz_2 - raiz_1)
    
    return (b1,b2)

def funciones_bases(n,x):
    
    b = bases_cardinales(n,x)
    b1 = b[0]
    b2 = b[1]
    
    f1 = (np.exp(-x))*b1
    f2 = (np.exp(-x))*b2
    
    return (f1,f2)

def pesos_cardinales(n,x):
    
    b = bases_cardinales(n,x)
    b1 = b[0]
    b2 = b[1]
    
    valor = 0
    
    peso_1 = (sym.exp(-x))* b1
         
    I = ((limite_1 - limite_0)/2)*valor
    
    return I
    
#punto_d

def GetAllRootsGLag2(n):
    
    c_s = n+(n-1)*np.sqrt(n)
    
    xn = np.linspace(0,c_s,1000)
    
    Laguerre = []
    DLaguerre = []
    
    for i in range(n+1):
        Laguerre.append(GetLaguerre(i,x1))
        DLaguerre.append(GetDLaguerre(i,x1))
    
    poly = sym.lambdify([x1],Laguerre[n],'numpy')
    Dpoly = sym.lambdify([x1],DLaguerre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots

def GetWeightsGLag(n):
    
    Weights = np.array([])
    Roots = GetAllRootsGLag2(n)
    
    for i in Roots:
        
        Weights_t = i/(((n+1)**2)*((GetLaguerre(n+1,i))**2))
        Weights = np.append(Weights,Weights_t)
    
    return (Weights, Roots)

def x_3(x):
    
    f = x**3

    return f

def Integral_Cuadratura_GLag(n):

    valor = 0
    w = GetWeightsGLag(n)
    pesos = w[0]
    raices = w[1]
    
    for i in range(n):
        
        valor += pesos[i]*x_3(raices[i])
        
    return valor

#valor_igualdad_6 = Integral_Cuadratura_GLag(3)
#print(valor_igualdad_6)

