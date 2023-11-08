import numpy as np
import sympy as sym
import matplotlib.pyplot as plt



x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)



#Punto 4

M1 = np.matrix('1 0 0; 5 1 0; -2 3 1')
M2 = np.matrix('4 -2 1; 0 3 7; 0 0 2')

def multiplicacion_matricial(M1, M2):
    
    if M1.shape[1] == M2.shape[0]:
        
        M12 = np.zeros((M1.shape[0],M2.shape[1]), np.int64)
        n_r = M1.shape[0]
        n_c = M2.shape[1]
        n_e = M1.shape[1]
            
        for i in range(n_r):
    
            r = M1[i,:]
            
            for j in range(n_c):
                
                c = M2[:,j]
                s = 0
                
                for k in range(n_e):
                
                    m = r[0,k]*c[k,0]
                    s += m
            
                M12[i,j] = s
            
        return M12
        
    else:
        
        print('El número de columnas de la primera matriz no es igual al numero de filas de la segunda.')

#M12 = multiplicacion_matricial(M1, M2)
#print(M12)



#punto_17

def f_17(x,y):
    
    z = x + sym.I*y
    f = z**3 - 1
    f = f.expand()
    return sym.re(f),sym.im(f)

f0,f1 = f_17(x,y)
F = [f0,f1]

def Jacobian(F,x,y):

    J = sym.zeros(2,2)
    
    for i in range(2):
        for j in range(2):
            
            if j==0:
                
                J[i,j] = sym.diff(F[i],x,1)
                
            else:
                
                J[i,j] = sym.diff(F[i],y,1)
                
    return J

J_j = Jacobian(F,x,y)
InvJ = J_j.inv()

Fn = sym.lambdify([x,y],F,'numpy')
IJn = sym.lambdify([x,y],InvJ,'numpy')

def NewtonRaphson(Fn,Jn,z,itmax=300,precision=1e-9):
    
    error = 1
    it = 0
    
    while error > precision and it < itmax:
        
        IFn = Fn(z[0],z[1])
        IJn = Jn(z[0],z[1])
        
        z1 = z - np.dot(IJn,IFn)
        
        error = np.max(np.abs(z1-z))
        
        z = z1
        it +=1
        
    zz = (z[0],z[1])
        
    return zz

x0 = np.linspace(-1, 1, 300)
y0 = np.linspace(-1, 1, 300)

def todas_raices(Fn,Jn):
    
    raices = []
    tolerancia = 7
    puntos_raiz = []
    k = 0
    for i in x0:
        
        w = 0
        
        for j in y0:
            
            raiz = NewtonRaphson(Fn,Jn,[i,j])
            craiz0 = np.round(raiz[0],tolerancia)
            craiz1 = np.round(raiz[1],tolerancia)
            craiz = (craiz0,craiz1)
            pr = ((k,w),craiz)
            puntos_raiz.append(pr)
            
            if craiz not in raices:
                
                raices.append(craiz)
                
            w += 1
            
        k += 1
                
    raices.sort()
        
    return raices,puntos_raiz

#z_r = todas_raices(Fn,IJn)

def graficar_fractal():
    
    fractal = np.zeros((300,300), np.int64)
    puntos = z_r[1]
    
    for i in puntos:
        
        j = i[0]
        k = i[1]
        
        if k[1] == -0.8660254:
            
            fractal[j] = 20
        
        if k[1] == 0.8660254:
            
            fractal[j] = 100
            
        if k[1] == -0.0:
            
            fractal[j] = 255
    
    plt.imshow(fractal, cmap='coolwarm' ,extent=[-1,1,-1,1])

#graficar_fractal()