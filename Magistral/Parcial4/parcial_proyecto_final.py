import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

sigm = lambda x: 1/(1+np.exp(-x))

class Layer:
    
    def __init__(self,NC,NN,ActFun,rate=0.4): # Jugar con la tasa de mutacion
        
        self.NC = NC
        self.NN = NN
        self.ActFunc = ActFun
        self.rate = rate
        
        self.W = np.random.uniform( -10.,10.,(self.NC,self.NN) )
        self.b = np.random.uniform( -10.,10.,(1,self.NN) )
        
    def Activation(self,x):
        z = np.dot(x,self.W) + self.b
        return self.ActFunc(z)[0]
    
    def Mutate(self):
        
        self.W += np.random.uniform( -self.rate, self.rate, size=(self.NC,self.NN))
        self.b += np.random.uniform( -self.rate, self.rate, size=(1,self.NN))
        
def GetBrain():
    
    l0 = Layer(1,5,sigm)
    l1 = Layer(5,1,sigm)
    Brain = [l0,l1]
    
    return Brain    

class Robot:
    
    def __init__(self, dt, Layers, Id=0):
        
        self.Id = Id
        self.dt = dt
        
        self.r = np.random.uniform([0.,0.])
        theta = 0.
        self.v = np.array([1.*np.cos(theta),1.*np.sin(theta)])

        
        # Capacidad o aptitud del individuo
        self.Fitness = np.inf
        self.Steps = 0

        # Brain
        self.Layers = Layers
        
    def GetR(self):
        return self.r
    
    def Evolution(self):
        self.r += self.v*self.dt # Euler integration (Metodos 2)

        # Cada generación regreamos el robot al origin
        # Y volvemos a estimar su fitness
        
    def Reset(self):
        self.Steps = 0.
        self.r = np.array([0.,0.])
        self.fitness = np.inf
        
    # Aca debes definir que es mejorar en tu proceso evolutivo
    
    def SetFitness(self):
        
        d = 1/self.Steps 
        
        self.Fitness = d
        
       # Brain stuff
    def BrainActivation(self,x,threshold=0.7): 
        # El umbral (threshold) cerebral es a tu gusto!
        # cercano a 1 es exigente
        # cercano a 0 es sindrome de down
        
        # Forward pass - la infomación fluye por el modelo hacia adelante
        for i in range(len(self.Layers)):         
            if i == 0:
                output = self.Layers[i].Activation(x)
            else:
                output = self.Layers[i].Activation(output)
        
        self.Activation = np.round(output,4)
        
        # Cambiamos el vector velocidad
        if self.Activation[0] > threshold:
            self.v = -self.v
            self.Steps += np.random.uniform(-0.9,0.)
    
        return self.Activation
    
    # Aca mutamos (cambiar de parametros) para poder "aprender"
    def Mutate(self):
        for i in range(len(self.Layers)):
            self.Layers[i].Mutate()
    
    # Devolvemos la red neuronal ya entrenada
    def GetBrain(self):
        return self.Layers
    
def GetRobots(N):
    
    Robots = []
    
    for i in range(N):
        
        Brain = GetBrain()
        r = Robot(dt,Brain,Id=i)
        Robots.append(r)
        
    return Robots

dt = 0.1
t = np.arange(0.,15.1,dt)

#Recordemos que son 200 robots los que estan siendo evaluados, aunque solo se muestran 5.
#No necesariamente uno de estos 5 es el mejor, el mejor se elige al final.

def GetPlot():
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
 
    return ax,ax1

def TimeEvolution(Robots,e,best_fitness,Plot=True):
    
    for it in range(t.shape[0]):
        
        if Plot:
        
            clear_output(wait=True)
        
            ax,ax1 = GetPlot()
            ax1.set_ylim(0.,1.)
        
            ax.set_title('t = {:.3f}'.format(t[it]))
        
        Activation = np.zeros(len(Robots))
        
        for i,p in enumerate(Robots):
            p.Evolution()
         
            # Activacion cerebral
            
            Act = p.BrainActivation(p.GetR()[0] - best_fitness)
            Activation[i] = Act
               
            # Region donde aumentamos los pasos para el fitness
            
            if p.r[0] < 1:
                
                if p.r[0] > -1:
                    
                    p.Steps += np.random.uniform(0.,0.9)
                
            if Plot and i < 5: # Solo pintamos los primeros 5, por tiempo de computo
                ax.scatter(p.r[0],p.r[1],label='Id: {}, Steps: {:.0f}'.format(p.Id,p.Steps))
                ax.quiver(p.r[0],p.r[1],p.v[0],p.v[1])
                
        # Pintamos la activaciones de los primeros 5
        
        if Plot:
            ax1.plot(np.arange(0,len(Robots[:5]),1),Activation[:5],marker='o',color='b',label='Activation')
            ax1.axhline(y=0.7,color='r')
        
        if Plot:
        
            ax.legend(loc=0)  
            ax1.legend(loc=0)
            plt.show()
            time.sleep(0.001)
            
# Definimos la rutina de entrenamiento
#Solo 10 epochs, si lo hacemos con más se demora demasiado.
def Genetic(epochs = 10, Plot = True, Plottime=False):
    
    N = 200
    Robots = GetRobots(N)
    
    FitVector = np.array([])
    best_fitness = 0
        
    x = np.linspace(-1,1,20)
    Act_0 = np.zeros_like(x)
    
    for e in range(int(epochs)):
        
        # Reiniciamos y mutamos los pesos
        
        for p in Robots:
            p.Reset() 
            p.Mutate()
            
        # Evolucionamos
        TimeEvolution(Robots,e,best_fitness,Plottime) # Apagar dibujar la evolución para entrenar
        
        # Actualizamos fitness de cada robot
        
        pos = False
        for p in Robots:
            
            p.SetFitness()
        
        # Aca va toda la rutina de ordenar los bots del más apto al menos apto
        
            scores = [(p.Fitness,p) for p in Robots]
            scores.sort(key = lambda x: x[0], reverse = False)
            
        while pos == False:
                
            if scores[0][0] < 0:
                    
                scores.pop(0)
                
            else:
                
                pos = True
        
        # Guardamos el mejor fitness y el mejor robot
        
        best_fitness = scores[0][0]
        best_bot = scores[0][1]
        
        FitVector = np.append(FitVector,best_fitness)
        
        for i in range(len(x)):
            Act_0[i] = best_bot.BrainActivation(x[i])
        
        clear_output(wait=True)
        
        print('Epoch:', e)
        print('Last Fitness:', FitVector[-1])        
        
        if Plot:
            
            ax,ax1 = GetPlot()
            ax.plot(x,Act_0,color='k')
            ax.set_ylim(0.,1)
            ax.axhline(y=0.7,ls='--',color='r',label='Threshold')
            
            ax1.set_title('Fitness')
            ax1.plot(FitVector)
        
            ax.legend(loc=0)
            
            plt.show()
            
            time.sleep(0.001)
    
    return best_bot, FitVector

Best, FitVector = Genetic(Plot=True,Plottime=True) # Apagar Plottime para el entrenamiento
#plt.plot(FitVector)

#Mostramos el movimiento del mejor robot luego de 10 epochs

def GetPlotBest():
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
 
    return ax

def TimeEvolutionBest(robot):
    
    robot.r = np.array([0.,0.])
    for it in range(t.shape[0]):
        
        clear_output(wait=True)
        
        ax = GetPlotBest()
        ax.set_title('t = {:.3f}'.format(t[it]))
        
        robot.Evolution()
         
        Act = robot.BrainActivation(robot.GetR()[0])
        
        ax.scatter(robot.r[0],robot.r[1],label='Id: {}, Brain Activation: {:.2f}'.format(robot.Id,Act[0]))
        ax.quiver(robot.r[0],robot.r[1],robot.v[0],robot.v[1])
        
        ax.legend()
        plt.show()
        
        time.sleep(0.001)

TimeEvolutionBest(Best)