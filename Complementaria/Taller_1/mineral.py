# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:14:27 2023

@author: elect
"""

import matplotlib.pyplot as plt

class Mineral:
    
    #Punto 1.1

    def __init__(self,nombre:str,dureza:float,lustre:str,rompimiento_por_fractura:bool,color:str,
                 composición:str,sistema_cristalino:str,specific_gravity:float):
        
        self.nombre = nombre
        self.dureza = dureza
        self.lustre = lustre
        self.rompimiento_por_fractura = rompimiento_por_fractura
        self.color = color
        self.composicion = composición
        self.sistema_cristalino = sistema_cristalino
        self.gravedad_especifica = specific_gravity
        
    #Punto 1.2
            
    def clasificacion_silicato(self):
        
        composicion = self.composicion
        j = 0
        n = 0
        i = 0
        Si_encontrado = False
        O_encontrado = False
        silicato = False
        
        while i < len(composicion):
            
            if Si_encontrado == False:
        
                if composicion[i] == 'S':
                
                    j = i + 2
                
                    if composicion[i:j] == 'Si':
                    
                        if not composicion[j].islower():
                        
                            n += 1
                            Si_encontrado = True
                            
            if O_encontrado == False:
                        
                if composicion[i] == 'O':
                
                    j = i + 2
                
                    if not composicion[i+1:j].islower():
                    
                        if not composicion[i+1:j] == 'H':
                    
                            n += 1
                            O_encontrado = True
                    
            if n == 2:
            
                silicato = True
                i = len(composicion)
                
            else:
                
                i += 1
            
        return silicato
        
    def densidad_SI(self):
        
        gravedad_esp = self.gravedad_especifica
        d_agua_4C = 1000    
        d_material = gravedad_esp*d_agua_4C
        str_d = f'{d_material} kg/m^3'
        
        return str_d
    
    def visualizacion_color_material(self):
        
        figure = plt.Rectangle((1,1),1,1,facecolor=self.color,edgecolor=self.color)
        ax = plt.gca()
        ax.add_patch(figure)
        plt.axis('scaled')
        plt.axis('off')
        plt.show()
        
    def imprimir_dureza_rompimiento_sistema(self):
        
        if self.rompimiento_por_fractura == True:
            
            rompimiento = 'Por fractura'
            
        else:
            
            rompimiento = 'Por escisión'
        
        print(f'Dureza: {self.dureza}\nTipo de Rompimiento: {rompimiento}\nSistema de organización atomico: {self.sistema_cristalino}')

'''    
Noco = Mineral('e',1,'e',True,'#7a5596','Mg3Si4O10(OH)2','Isométrico',5.3)

print(Noco.clasificacion_silicato(),',',Noco.densidad_SI(),'\n')

Noco.imprimir_dureza_rompimiento_sistema()

Noco.visualizacion_color_material()'''