# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:52:06 2023

@author: elect
"""

from mineral import Mineral
import numpy as np
import matplotlib.pyplot as plt

class ExpansionTermicaMineral(Mineral):
    
    def __init__(self,nombre:str,dureza:float,lustre:str,rompimiento_por_fractura:bool,color:str,
                 composición:str,sistema_cristalino:str,specific_gravity:float,ruta_csv:str,lista_temperatura:list,lista_volumen:list):
        
        super().__init__(nombre,dureza,lustre,rompimiento_por_fractura,color,composición,sistema_cristalino,specific_gravity)
        
        self.ruta = ruta_csv
        
        lista_temperatura = []
        lista_volumen = []
        
        archivo = open(self.ruta, 'r')
        lineas = archivo.readlines()
        
        for i in lineas[1:]:
            
            datos = i.strip().split(',')
            
            lista_temperatura.append(float(datos[0]))
            lista_volumen.append(float(datos[1]))
        
        self.temperatura = lista_temperatura
        self.volumen = lista_volumen
        
    def calcular_y_visualizar_coeficiente_expansion_termica(self):
        
        tupla_coeficiente_y_error = ()
        temp_np = np.asarray(self.temperatura)
        volumen_np = np.asarray(self.volumen)
        
        h = temp_np[1] - temp_np[0]
        d = ((volumen_np + h) - (volumen_np - h)) / (2*h)
        
        coeficiente = (1/volumen_np)*d
        error = np.abs(volumen_np - d)
        
        tupla_coeficiente_y_error = (coeficiente,error)
        
        fig = plt.figure(figsize=(15,7))
        plt.suptitle(f'Expansión Termica del mineral {self.nombre}', fontsize = 20)
        ax_volumen = fig.add_subplot(121)
        ax_volumen.scatter(self.temperatura,self.volumen)
        ax_volumen.set_xlabel('Temperatura (T=°C)', fontsize = 12)
        ax_volumen.set_ylabel('Volumen (V = cm^3)', fontsize = 12)
        ax_volumen.set_title('Volumen en función de la Temperatura', fontsize = 15)
        
        ax_coeficiente = fig.add_subplot(122)
        ax_coeficiente.scatter(self.temperatura,coeficiente, color = 'orange')
        ax_coeficiente.set_xlabel('Temperatura (T=°C)', fontsize = 12)
        ax_coeficiente.set_ylabel('Coeficiente (α)', fontsize = 12)
        ax_coeficiente.set_title('Coeficiente en función de la Temperatura', fontsize = 15)
        
        return tupla_coeficiente_y_error

#olivino = ExpansionTermicaMineral('e', 1, 'e', True, 'r', 'Bf3', 'e', 2.8, 'graphite_mceligot_2016.csv',[],[])
#olivino.calcular_y_visualizar_coeficiente_expansion_termica()

        