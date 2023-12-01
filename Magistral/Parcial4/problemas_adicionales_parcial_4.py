import numpy as np
import sympy as sys
import matplotlib.pyplot as plt
import scipy.optimize as spy

xsys = sys.Symbol('x',real=True)
ysys = sys.Symbol('y',real=True)

"""
Recuerden quitar los '#' de ciertas variables donde se guardan los resultados de funciones para que el resto funcionen.

Ej:
    
    #s = prob_cumple_diferente(365)
    #sample = get_sample_p8()
    #l_ST_prob = prob_h1_inicial()
    
"""





#Optimización:



#Punto 2

def fop2(x):
    
    x1 = x[0]
    x2 = x[1]
    
    f = ((29/25)*(x1**2) + (41/25)*(x2**2) -(16/25)*x1*x2 + (12/25)*x1 - (24/25)*x2 + (9/25))

    return f

x0y0po2 = [1,1]
min_fop2 = spy.minimize(fop2, x0y0po2)
#print(min_fop2.fun)



#Punto 3

#c&d

def fop3(x,s=-1):
    
    x1 = x[0]
    x2 = x[1]
    
    f = ((12*x1*x2 -(x1**2)*(x2**2))/(2*(x1+x2)))
    _f = s*f

    return _f

#Para maximizar debemos multiplicar la funcion por -1, debido a que la minimización de -f es igual a la maximización de f.

x0y0po3 = [1,1]
max_vop3 = spy.minimize(fop3, x0y0po3)
#print(max_vop3.fun)

#El resultado nos da -4cm^3. Basado en la explicación previa, el valor máximo de volumen es de 4 cm^3.





#Generales de probabilidad:



#Punto 4

def prob_cumple_diferente(n):
    
    multi = 1
    F_P = []
    
    for i in range(0,n):
        
        p = np.abs(i - 365)
        caso = p/365
        multi *= caso
        
        prob = multi * 100
        F_P.append(prob)
        
    return (prob,F_P)

#s = prob_cumple_diferente(365)

def graficar_f_prob_n():
    
    prob = s[1]
    prob_y = prob[:80]
    n_x = np.linspace(1,80,80)
    plt.scatter(n_x,prob_y)
    plt.xlabel('Número de personas (n)')
    plt.ylabel('Probabilidad de cumpleaños en diferentes días (%)', fontsize=8.75)
    
#graficar_f_prob_n()



#Punto 8

def get_sample_p8(N = int(1e5), ncoins = 4, Weights=None):
    
    sample = np.zeros((N,ncoins))
    events = [1,-1]
    
    for i in range(N):
        
        if Weights == None:
            Exp = np.random.choice(events,ncoins)
            
        sample[i] = Exp
        
    return sample

#sample = get_sample_p8()

def dos_dos():

    n_dos_dos = 0
    
    for i in range(sample.shape[0]):
    
        suma = 0
    
        for j in range(sample.shape[1]):
        
            suma += sample[i,j]
            
        if suma == 0:
            
            n_dos_dos += 1
            
    prob = (n_dos_dos/(1e5)) * 100
        
    return prob
            
#prob_p8 = dos_dos()
#print(prob_p8)

#Como podemos notar, la probabilidad de obtener 2 caras y 2 sellos siempre esta cerca del valor 3/8 * 100% = 37.5% 





#Distribuciones continuas de probabilidad:

    

#Punto 1








#Hidden Markov Models: 
    
    

#Punto 1



#a & b

apriori = (0.2,0.8)
T = np.matrix('0.8 0.2; 0.2 0.8')
E = np.matrix('0.5 0.9; 0.5 0.1')
SO_inicial = np.array([1,0,0,0,1,0,1,0]) # C=0, S=1

def crear_SH_inicial():
    
    l_ST = []
    SO = SO_inicial
    
    while len(l_ST) < 256:
    
        SH = np.random.randint(0, 2, size=8) #J=0, B=1   
        ST = np.array([])
        ST = np.stack((SH,SO))
        
        if len(l_ST) == 0:
                
            l_ST.append(ST)
            
        else:
            
            i = 0
            
            while i < len(l_ST):
            
                SH_c = ST[0]    
                ST_c2 = l_ST[i] 
                SH_c2 = ST_c2[0]
                dif = 0
                
                for j in range(0,8):
                    
                    if SH_c[j] != SH_c2[j]:
                        
                        dif += 1
                        
                if dif == 0:
                
                    i = len(l_ST) + 1
                    
                else:
                    
                    i += 1
                    
            if i == len(l_ST):
                
                l_ST.append(ST)
    
    return l_ST

def prob_h1_inicial():
    
    l_ST = crear_SH_inicial() 
    l_ST_prob = []
    
    for k in range(0,len(l_ST)):
        
        ST = l_ST[k]
        prob = 1
        
        for i in range(0,ST.shape[1]):
        
            if prob == 1:
            
                estadoSH = ST[0,i]
                estadoSO = ST[1,i]
        
                prob *= (E[estadoSO,estadoSH]*apriori[estadoSH])
            
            else:
            
               estadoSH = ST[0,i]
               estadoSH_a = ST[0,i-1]
               estadoSO = ST[1,i]
            
               prob *= (E[estadoSO,estadoSH]*T[estadoSH,estadoSH_a])
               
        l_ST_prob.append((ST[0],(prob*100)))
            
    return l_ST_prob

#l_ST_prob = prob_h1_inicial()

def prob_mayor_h1_inicial():

    prob_m = 0.0002
    
    for i in range(0,len(l_ST_prob)):
        
        ST_prob = l_ST_prob[i]
        
        if ST_prob[1] > prob_m:
            
            prob_m = ST_prob[1]
            SH_m = ST_prob[0]
            
    return (SH_m,prob_m)

#SH_prob_m_inicial = prob_mayor_h1_inicial()

#La secuencia oculta/hidden con mayor probabilidad es 'B B B B J J J J' con una probabilidad del 0.019%



#c

def prob_SO_inicial():
    
    SO_prob = 0
    
    for i in range(0,len(l_ST_prob)):
        
        ST_prob = l_ST_prob[i]
        SO_prob += ST_prob[1]

    return SO_prob

#prob_SO_inicial = prob_SO_inicial()

#La probabilidad de la secuencia inicial es del 0.1934%



#d

def Crear_SO():
    
    l_SO = []
    
    while len(l_SO) < 256:
    
        SO = np.random.randint(0, 2, size=8) #J=0, B=1   
        
        if len(l_SO) == 0:
                
            l_SO.append(SO)
            
        else:
            
            i = 0
            
            while i < len(l_SO):
            
                SO_c = SO    
                SO_c2 = l_SO[i] 
                dif = 0
                
                for j in range(0,8):
                    
                    if SO_c[j] != SO_c2[j]:
                        
                        dif += 1
                        
                if dif == 0:
                
                    i = len(l_SO) + 1
                    
                else:
                    
                    i += 1
                    
            if i == len(l_SO):
                
                l_SO.append(SO)
    
    return l_SO

#l_SO = Crear_SO()

#Las 256 posibles secuencias ocultas/hidden de una secuencia observable son las mismas para cualquier otra.

def Crear_SH():
    
    l_SH = []
    
    while len(l_SH) < 256:
    
        SH = np.random.randint(0, 2, size=8) #J=0, B=1   
        
        if len(l_SH) == 0:
                
            l_SH.append(SH)
            
        else:
            
            i = 0
            
            while i < len(l_SH):
            
                SH_c = SH    
                SH_c2 = l_SH[i] 
                dif = 0
                
                for j in range(0,8):
                    
                    if SH_c[j] != SH_c2[j]:
                            
                        dif += 1
                        
                if dif == 0:
                
                    i = len(l_SH) + 1
                    
                else:
                    
                    i += 1
                    
            if i == len(l_SH):
                
                l_SH.append(SH)
    
    return l_SH

def prob_h1():

    l_SH = Crear_SH() 
    l_SA_prob= []
    
    for k in range(0,len(l_SH)):
        
        SO = l_SO[k]
        l_ST_prob = []
        
        for j in range (0,len(l_SH)):
            
            SH = l_SH[j]
            ST = np.stack((SH,SO))
            prob = 1
        
            for i in range(0,ST.shape[1]):
        
                if prob == 1:
            
                    estadoSH = ST[0,i]
                    estadoSO = ST[1,i]
        
                    prob *= (E[estadoSO,estadoSH]*apriori[estadoSH])
            
                else:
            
                    estadoSH = ST[0,i]
                    estadoSH_a = ST[0,i-1]
                    estadoSO = ST[1,i]
            
                    prob *= (E[estadoSO,estadoSH]*T[estadoSH,estadoSH_a])
               
            l_ST_prob.append((ST[0],(prob*100)))
        
        l_SA_prob.append(l_ST_prob)
            
    return l_SA_prob

def prob_SO_total():
    
    SO_prob = 0
    l_SA_prob = prob_h1()
    
    for i in range(0,len(l_SA_prob)):
        
        ST_prob = l_SA_prob[i]
        
        for j in range(0,len(ST_prob)):
        
            SH_prob = ST_prob[j]
            SO_prob += SH_prob[1]

    return SO_prob

#SO_total = prob_SO_total()

"""
Como podemos observar, la sumatoria de todos los estados observables, el cual es igual
a la sumatoria de todas las secuencias observables es igual a 100%, el cual es 1.
"""



#e

"""
Claramente, al ajustar los valores de probabilidad a-priori estamos cambiando las probabilidades iniciales que J & B toman.
Por lo que las probabilidades de cada secuencia oculta/hidden y observable cambian.

Con el apriori original de [0.2,0.8], la probabilidad de la secuencia observable inicial es del 0.1934%
Con [0.3,0.7], la probabilidad se convierte en 0.2219%
Con [0.5,0.5], es 0.2789%

y asi...

Como la probabilidad de la secuencia observable depende de las probabilidades de las secuencias ocultas/hidden. Estas también cambian.
"""





#Mínimos cuadrados:



#Punto 1b

def tres_lineas_punto():
    
    x = np.linspace(-5,5,1000)
    y = np.zeros((3,len(x)))
    
    y[0] = 2*x - 2
    y[1] = -0.5*x + 0.5
    y[2] = 4 - x
    
    A = np.array([[2,-1],[1,2],[1,1]])
    b = np.array([2,1,4])
    M = np.dot(A.T,A)
    bt = np.dot(A.T,b)
    xsol = np.linalg.solve(M,bt)
    
    return (xsol,(x,y))

def graficar_Mc1():
    
    valores = tres_lineas_punto()
    y_v = valores[1]
    xsol = valores[0]
    
    y = y_v[1]
    x = y_v[0]
    
    fig = plt.figure()
    a = fig.add_axes([0,0,1,1])
    a.set_ylim(-5,5)
    a.set_xlim(-5,5)
    a.scatter(xsol[0],xsol[1],color='r')
    
    for l in range(y.shape[0]):
    
        a.plot(x,y[l],ls='--',lw=2)
    
#graficar_Mc1() 

#Punto 7

#b

A7 = np.array([[3,1,-1],[1,2,0],[0,1,2],[1,1,-1]])
b7 = np.array([-3,-3,8,9])

def grand_schmidt():
    
    m0 = A7[:,0]
    M = np.zeros((A7.shape[0],A7.shape[1]))
    
    M[:,0] = m0
    
    for i in range(1,M.shape[1]):
        
        m = A7[:,i]
        suma = 0
        
        for j in range(0,M.shape[1]-1):
            
            deno = np.dot(M[:,j],M[:,j])
            
            if deno == 0:
                
                suma += -(np.dot(M[:,j],m)/1)*M[:,j]
            
            else:
                
                suma += -(np.dot(M[:,j],m)/np.dot(M[:,j],M[:,j]))*M[:,j]
            
        mf = m + suma
        M[:,i] = mf
    
    return M

def convertir_ortonormales():
    
    M = grand_schmidt()
    V = np.zeros((M.shape[0],M.shape[1]))
    
    for i in range(0,V.shape[1]):
        
        v = 1/(np.linalg.norm(M[:,i]))
        vf = np.dot(v,M[:,i])
        V[:,i] = vf
        
    return V

def hallar_constantes():
    
    V = convertir_ortonormales()
    l_c = []
    
    for i in range(0,V.shape[1]):
    
        c = np.dot(b7,V[:,i])/np.dot(V[:,i],V[:,i])
        l_c.append(c)
        
    return (l_c,V)

def hallar_proyeccion():
    
    L = hallar_constantes()
    l_c = L[0]
    V = L[1]
    proy = 0
    
    for i in range(0,len(l_c)):
        
        proy += l_c[i]*V[:,i]
        
    return proy

#proy_b7 = hallar_proyeccion()

#La proyección ortogonal b es claramente [-2,3,4,0]



#a

def gauss_jordan():
    
    proy_b = proy_b7
    A = A7
    
    M = np.zeros([A.shape[0],A.shape[1]+1])
    
    for i in range(0,A.shape[0]):
        
        f1 = A[i]
        f2 = proy_b[i]
        
        f1 = np.append(f1,f2)
        
        M[i] = f1
        
    #M = np.array([[4,-2,-1,1,2,14],[1,2,2,-1,4,14],[2,-1,4,-2,2,-8],[1,1,1,1,1,23],[6,4,1,-6,6,-4]],dtype='float64')
    #M = np.array([[3,2,1,-1,3],[1,1,-2,-3,5],[7,8,7,3,2]],dtype='float64')
    
    """
    Nuestra matriz original es de 4x3, 4x4 si juntamos el b. Debido a ello debemos usar el algoritmo siguiente
    Si hubiera sido 4x4 -> 4x5 hubieramos tenido que usar el algoritmo que le sigue a este.
    De la misma manera si hubiera sido 3x4 -> 3x5 hubieramos tenido que usar el ultimo algoritmo.
    """   
    
    if (M.shape[0] - M.shape[1]) == 0:
     
        n = 0
        
        while n < (M.shape[1]-1):
            
            for i in range(0,M.shape[0]-1):
                
                va = M[i,i]
                n += 1
                
                if va != 1:
                      
                    vaa = 1/va
                    mod = np.dot(vaa,M[i])
                    M[i] = mod
                    va = M[i,i]
                    
                for j in range(0,M.shape[1]):
                
                    vb = M[j,i]
                    
                    if vb != 0:
                        
                        if (j+i) != (i+i) :
                    
                            f = -(vb*va)
                        
                            for k in range(0,M.shape[1]):
                            
                                M[j,k] = M[j,k] + f*M[i,k]
                                
                                
        N = M[:,0:M.shape[1]-1]         
        suma = np.sum(N,axis=1)         
                      
        for n in range(0,M.shape[1]):
                
            if suma[n] == 0:
                    
                M = np.delete(M,n,axis=0)
                
                
                                
    if (M.shape[0] - M.shape[1]) == -1:
        
        n = 0
        
        while n < (M.shape[1]-1):
            
            for i in range(0,M.shape[0]):
                
                va = M[i,i]
                n += 1
                
                if va != 1:
                      
                    vaa = 1/va
                    mod = np.dot(vaa,M[i])
                    M[i] = mod
                    va = M[i,i]
                    
                for j in range(0,M.shape[0]):
                
                    vb = M[j,i]
                    
                    if vb != 0:
                        
                        if (j+i) != (i+i) :
                    
                            f = -(vb*va)
                        
                            for k in range(0,M.shape[1]):
                            
                                M[j,k] = M[j,k] + f*M[i,k]
    
                
        N = M[:,0:M.shape[1]-1]         
        suma = np.sum(N,axis=1)         
              
        for n in range(0,M.shape[0]):
        
            if suma[n] == 0:
            
                M = np.delete(M,n,axis=0)
                
                
                
    if (M.shape[0] - M.shape[1]) == -2:
        
        n = 0
        
        while n < (M.shape[1]-2):
            
            for i in range(0,M.shape[0]):
                
                va = M[i,i]
                n += 1
                
                if va != 1:
                      
                    vaa = 1/va
                    mod = np.dot(vaa,M[i])
                    M[i] = mod
                    va = M[i,i]
                    
                for j in range(0,M.shape[0]):
                
                    vb = M[j,i]
                    
                    if vb != 0:
                        
                        if (j+i) != (i+i) :
                    
                            f = -(vb*va)
                        
                            for k in range(0,M.shape[1]):
                            
                                M[j,k] = M[j,k] + f*M[i,k]
    
                
        N = M[:,0:M.shape[1]-1]         
        suma = np.sum(N,axis=1)         
              
        for n in range(0,M.shape[0]):
        
            if suma[n] == 0:
            
                M = np.delete(M,n,axis=0)

    x7 = M[:,M.shape[1]-1]

    return x7

    
#x7 = gauss_jordan()

def verificar_x():
    
    proy_b = np.dot(A7,x7)
    
    if np.sum(proy_b) == np.sum(proy_b7):
        
        print("\nHemos encontrado la solución de mínimos cuadrados")
        
    else:
        
        print("\nEsa no es la solución, intenta de nuevo.")

#verificar_x()

#El x que hallamos es la solución de minimos cuadrados para la proyección encontrada previamente