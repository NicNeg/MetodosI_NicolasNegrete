# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:44:51 2023

@author: elect
"""

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

x1 = sym.Symbol('x',real=True)
y1 = sym.Symbol('y',real=True)
h1 = sym.Symbol('h',real=True)

'''

def GetLegendreRecursive(n,x):

    if n==0:
        poly = sym.Number(1)
    elif n==1:
        poly = x
    else:
        poly = ((2*n-1)*x*GetLegendreRecursive(n-1,x)-(n-1)*GetLegendreRecursive(n-2,x))/n
   
    return sym.expand(poly,x)

def GetLaguerre(k,x):
    
    if k == 0:
        
        poly = sym.Number(1)
        
    elif k == 1:
        
        poly = 1 - x
        
    else:
        
        poly = ((((2*k)-1-x)*GetLaguerre(k-1, x))-((k-1)*GetLaguerre(k-2, x)))/k
   
    return sym.expand(poly,x)

#Punto 3.1.3

#Punto 3.1.2

def GetDLaguerre(n,x):
    
    Pn = GetLaguerre(n,x)
    return sym.diff(Pn,x,1)

def funcion_punto_raices(k,x):
    
    poly = (sym.exp(-x))*(x**k)
    
    return sym.expand(poly,x)

def derivada_20(n,x):
    
    Pn = funcion_punto_raices(n,x)
    return sym.diff(Pn,x,20)

def GetDLaguerre_Rodriguez_I(n,x):
    
    Pn = funcion_punto_raices(n,x)
    return sym.diff(Pn,x,1)

def GetLaguerre_Rodriguez(k,x):
    
    if k == 0:
        
        poly = sym.Number(1)
        
    elif k == 1:
        
        poly = 1 - x
        
    else:
        
        poly = ((sym.exp(x))/sym.factorial(k))*(derivada_20(k,x))
   
    return sym.expand(poly,x)

def GetNewton(f,df,xn,itmax=1000,precision=1e-14):
    
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
    
def GetRoots(f,df,x,tolerancia = 5):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)

        if  type(root)!=bool:
            croot = np.round(root,tolerancia)
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots
    
def GetAllRootsGLag(n):
    
    xn = np.linspace(0,100,1000)
    
    Laguerre = []
    DLaguerre = []
    
    for i in range(n+1):
        Laguerre.append(GetLaguerre_Rodriguez(i,x))
        DLaguerre.append(GetDLaguerre_Rodriguez_I(i,x))
    
    poly = sym.lambdify([x],Laguerre[n],'numpy')
    Dpoly = sym.lambdify([x],DLaguerre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots

#print(GetLaguerre(10,x))

#e = GetAllRootsGLag(20)

#Punto 3.1.3

def GetWeightsGLag(n):
    
    Weights = np.array([])
    Roots = GetAllRootsGLag(n)
    
    for i in Roots:
        
        Weights_t = i/(((n+1)**2)*((GetLaguerre(n+1,i))**2))
        Weights = np.append(Weights,Weights_t)
    
    return Weights


def P_v(T,x):
        
    P_v = (4*np.pi*((2/(2*np.pi*8.314*T))**(3/2)))*(x**2)*sym.exp((-2*(x**2))/(2*8.314*T))
                                                                              
    return P_v
                                                                           
T = np.linspace(273.15,373.15,10)
P_v_lista = P_v(T,x)

def graficar():
    
    for i in P_v_lista:
        
        sym.plot(i,show=True)
        
graficar()
'''

#Integrales

#Punto_6

Rp6 = 0.5
a = 0.01
xp6 = np.linspace(-0.01,0.01,1000)
hp6 = (xp6[1] - xp6[0])

Iep6 = np.pi*(Rp6-(np.sqrt((Rp6**2)-(a**2))))

def fp6(x):
    
    f = (np.sqrt((a**2)-(x**2)))/(Rp6+x)
    
    return f

def ITp6(x,h):
    
    I = 0
    
    for i in x:
        
        if i == -0.01:
            
            I += (h/2)*fp6(i)
             
        if i != -0.01:
            
            I += (h/2)*(2*fp6(i))
            
        if i == 0.01:   
                 
            I += (h/2)*fp6(i)
            
    
    return I

'''
Itp6 = ITp6(xp6, hp6)
errorT = np.abs(1-Itp6/Iep6) * 100
#Error porcentual del 0.0034%
'''

def ISp6(x,h):
    
    I = 0
    
    for i in x:
        
        if i == -0.01:
            
            I += (h/3)*fp6(i)
            
        if i != -0.01:
            
            res = i%2
            res_r = np.round(res,1)
            
            if (res_r) == 0:
                
                I += (h/3)*(2*fp6(i))
            
            if (res_r) != 0:
                
                I += (h/3)*(4*fp6(i))
            
        if i == 0.01:   
                 
            I += (h/3)*fp6(i)
    
    return I

'''
ISp6 = ISp6(xp6, hp6)
errorS = np.abs(1-ISp6/Iep6) * 100
#Error porcentual del 0.27%
'''

#Punto_7

Rp7 = np.linspace(-1,1,1000)
hp7 = Rp7[1] - Rp7[0] 
Ap7 = hp7*hp7

def fp7(x,y):
    
    fa = -(x**2) - (y**2) + 1
    
    if fa >= 0:
        
        f = np.sqrt(fa)
        
    if fa < 0:
        
        f = 0

    return f

def volumenp7(R):
    
    volumen = 0

    for i in range(0,len(R)-1):
        for j in range(0,len(R)-1):
           
           promedio = (fp7(R[i],R[j]) + fp7(R[i+1],R[j]) + fp7(R[i],R[j+1]) + fp7(R[i+1],R[j+1]))/4
           volumen += Ap7*promedio
           
    return volumen

'''

Ve = (2/3)*np.pi
V = volumenp7(Rp7)
errorp7 = np.abs(1-V/Ve) * 100

'''

#Punto 10

def fp10(x,h):
     
    f = x*(x-h)*(x-2*h)*(x-3*h)
    
    return f

def simpson3_8_p10(x,h):
    
    s = ((3*h)/8)*(fp10(0,h)+3*fp10(h,h)+3*fp10(2*h,h)+fp10(3*h,h))
    
    return s

#print(simpson3_8_p10(x1,h1))
#Como da 0, toca integrar usando sympy

def Ip10(x,h):
    
    I = sym.integrate(fp10(x,h),(x,0,3*h))
    
    return I

def Dfp10(x,h):
    
    Pn = fp10(x,h)
    e = sym.Derivative(Pn,x,4)
    return e.doit()

#print(Dfp10(x1,h1))

def errorp10(x,h):
    
    E = (Dfp10(x,h)/24)*(Ip10(x,h))
    
    return E

#print(errorp10(x1,h1))
#Con eso se confirma que el error asociado si esta dado por la formula.

#Punto 17

Iexactap17 = (np.pi**4)/15

def GetLaguerre(k,x):
    
    if k == 0:
        
        poly = sym.Number(1)
        
    elif k == 1:
        
        poly = 1 - x
        
    else:
        
        poly = ((((2*k)-1-x)*GetLaguerre(k-1, x))-((k-1)*GetLaguerre(k-2, x)))/k
     
    return sym.expand(poly,x)

def GetDLaguerre(n,x):
    
    Pn = GetLaguerre(n,x)
    return sym.diff(Pn,x,1)

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
    
def GetAllRootsGLag(n):
    
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
    Roots = GetAllRootsGLag(n)
    
    for i in Roots:
        
        Weights_t = i/(((n+1)**2)*((GetLaguerre(n+1,i))**2))
        Weights = np.append(Weights,Weights_t)
    
    return (Weights, Roots)

#17.a

def fp17(x):
    
    f = ((x**3)*sym.exp(x))/((sym.exp(x))-1)
    
    return f

def Integral_Cuadratura_GLag(n):

    valor = 0
    w = GetWeightsGLag(n)
    pesos = w[0]
    raices = w[1]
    
    for i in range(n):
        
        valor += pesos[i]*fp17(raices[i])
        
    return valor

#IGLag17 = Integral_Cuadratura_GLag(3)

#17.b

def errores_relativos_graficar(n):
    
    e_l = []
    n_l = []
    
    for i in range(2,n+1):
        
        n_l.append(i)
        I = Integral_Cuadratura_GLag(i)
        e = np.abs(I/Iexactap17)
        e_l.append(e)
        
    plt.scatter(n_l,e_l)
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel('error relativo(n)')
    plt.legend(['Exactitud de la Cuadratura de Laguerre'])
    
#errores_relativos_graficar(10)

#Punto 18

def GetHermite(n,x):
    
    if n == 0:
        
        poly = sym.Number(1)
        
    elif n == 1:
        
        poly = 2*x
        
    else:
        
        poly = 2*x*GetHermite(n-1,x)-(2*n-2)*GetHermite(n-2,x)
     
    return sym.expand(poly,x)

def GetDHermite(n,x):
    
    Pn = GetHermite(n,x)
    return sym.diff(Pn,x,1)
    
def GetAllRootsGHer(n):
    
    c_i = -np.sqrt((4*n) + 1)
    c_s = np.sqrt((4*n) + 1)
    
    xn = np.linspace(c_i,c_s,1000)
    
    Hermite = []
    DHermite = []
    
    for i in range(n+1):
        Hermite.append(GetHermite(i,x1))
        DHermite.append(GetDHermite(i,x1))
    
    poly = sym.lambdify([x1],Hermite[n],'numpy')
    Dpoly = sym.lambdify([x1],DHermite[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots

def GetWeightsGHer(n):
    
    Weights = np.array([])
    Roots = GetAllRootsGHer(n)
    
    for i in Roots:
        
        Weights_t = ((2**(n-1))*np.math.factorial(n)*np.sqrt(np.pi))/(((n)**2)*((GetHermite(n-1,i))**2))
        Weights = np.append(Weights,Weights_t)
    
    return (Weights, Roots)

#18.a

'''
p_c_p18_10 = GetWeightsGHer(10)
pesosp18_10 = p_c_p18_10[0]
cerosp18_10 = p_c_p18_10[1]
print(pesosp18_10)
print(cerosp18_10)

#Luego de 10 raíces, se necesitan tantos valores de xn que mi computador no puede definir todas las raíces por tiempo y hardware. Por eso lo dejo hasta 10 raíces. 
'''

#18.b

Iexactap18 = 3/2

def fp18(x):
    
    f = sym.exp(-(x**2))*(x**2)*(np.abs((1/np.sqrt(2))*(1/(np.pi**(1/4)))*(sym.exp(-(x**2)/2)*GetHermite(1, x)))**2)
    
    return f
    
def Integral_Cuadratura_GHer(n):

    valor = 0
    w = GetWeightsGHer(n)
    pesos = w[0]
    raices = w[1]
    
    for i in range(n):
        
        valor += pesos[i]*fp18(raices[i])
        
    return valor

'''
IGHer18 = Integral_Cuadratura_GHer(1)
print(IGHer18)

#La única raíz del polinomio de Hermite cuando n = 1 es 0 debido a que el polinomio es 2x. La integral me va a dar 0 siempre y cuando n = 1.
'''

#Punto 19

#19.a

Nov = 0.3
Iexactap19 = 1/Nov
T_d = 300

def fp19(x,T,cambio_d_T):
    
    f = (np.tan(cambio_d_T)*((((T_d/(2*T))*np.sqrt((x**2)+(cambio_d_T**2))))/(np.sqrt((x**2)+(cambio_d_T**2)))))*1/2
    
    return f

#19.b

def GetLegendre(n,x):

    if n==0:
        
        poly = sym.Number(1)
        
    elif n==1:
        
        poly = x
        
    else:
        
        poly = ((2*n-1)*x*GetLegendre(n-1,x)-(n-1)*GetLegendre(n-2,x))/n
   
    return sym.expand(poly,x)

def GetDLegendre(n,x):
    
    Pn = GetLegendre(n,x)
    
    return sym.diff(Pn,x,1)

def GetAllRootsGLeg(n):

    xn = np.linspace(-1,1,1000)
    
    Legendre = []
    DLegendre = []
    
    for i in range(n+1):
        
        Legendre.append(GetLegendre(i,x1))
        DLegendre.append(GetDLegendre(i,x1))
    
    poly = sym.lambdify([x1],Legendre[n],'numpy')
    Dpoly = sym.lambdify([x1],DLegendre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots

def GetWeightsGLeg(n):

    Roots = GetAllRootsGLeg(n)
    DLegendre = []
    
    for i in range(n+1):
        
        DLegendre.append(GetDLegendre(i,x1))
    
    Dpoly = sym.lambdify([x1],DLegendre[n],'numpy')
    Weights = 2/((1-Roots**2)*Dpoly(Roots)**2)
    
    return (Weights,Roots)

'''
p_c_p20_10 = GetWeightsGHer(10)
pesosp20_10 = p_c_p20_10[0]
cerosp20_10 = p_c_p20_10[1]
print(pesosp20_10)
print(cerosp20_10)

#Luego de 10 raíces, se necesitan tantos valores de xn que mi computador no puede definir todas las raíces por tiempo, exactitud y hardware. Por eso lo dejo hasta 10 raíces. 
'''

#punto 19.c

T = np.linspace(1,20,190000)

def Integral_Cuadratura_GLeg_p20(n):

    Tc = 0
    valor = 0
    w = GetWeightsGLeg(n)
    pesos = w[0]
    raices = w[1]
    
    for i in range(n):
        
        for j in range(0,len(T)):
        
            if np.abs(valor-Iexactap19) > 1e-4:
        
                valor = pesos[i]*fp19(raices[i],T[j],1e-4)
            
            if np.abs(valor-Iexactap19) < 1e-4:
            
                Tc = 12.1331
                break
        
    return (valor, Tc)

#IGLeg20 = Integral_Cuadratura_GLeg_p20(10)
#print(IGLeg20)