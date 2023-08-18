# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 19:17:39 2023

@author: elect
"""

from pathlib import Path
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def buscar_enlace ():
    
    ruta1 = Path.cwd() / 'prueba'
    
    lista_ruta1 = os.listdir(ruta1)
    
    print('Elija una carpeta: ' + '\n\n1. '+lista_ruta1[0]+'\n2. '+lista_ruta1[1]+'\n3. '+lista_ruta1[2])
        
    #carpeta = int(input("\n Escriba su opción: "))
    
    #if carpeta == 1:
        
    ruta2 = Path.cwd() / 'prueba' / lista_ruta1[0]
    lista_ruta2 = os.listdir(ruta2) 
    
    print('Elija un archivo: ')

def crear_enlace_material_dic (ruta:str) -> dict:
    
    dic = {}
    j = 0
    
    archivo = pd.read_csv(ruta)
    material = archivo['Material']
    enlace = archivo['Enlace']
    
    for i in enlace:
        
        dic[i] = material[j]
        j += 1
        
    return dic

def crear_list_tupla_onda_y_n(ruta:str) -> list:
    
    lista = []
    onda = None
    onda2 = None
    continuar = True
    k = 0
    n = None
    archivo = open(ruta, 'r', encoding = 'utf8')
    
    datos = archivo.readline()
    e = len(datos)
    
    while e != 12:
        
        datos = archivo.readline()
        e = len(datos)
        
    if e == 12:
        
        while continuar:
            
            datos = archivo.readline()
            e = len(datos)
            
            if all([datos != "  - type: tabulated k\n", datos != "\n", datos != "SPECS:\n", e != 0, datos != " "]):
    
                y = datos[8:e]
                yy = len(y) - 1
            
                for i in y:
            
                    if i == " ":
                
                        kk = len(onda2) + 1
                        onda3 = k
                
                    else:
                
                        if k == 0:
                    
                           onda2 = i
                           k = onda2
                    
                        else:
                       
                           onda2 = k + i
                           k = onda2
                    
                n2 = y[kk:yy]
        
                onda = float(onda3)
                n = float(n2)
                    
                tupla = (onda,n)
                lista.append(tupla)
                k = 0
            
            else:
                
                continuar = False
        
    return lista

#print(tuplas_vaina('Iezzi.yml'))

def k_n(ruta:str) -> str :
    
    if ruta == 'French.yml':
    
        nombre = 'Kapton'
        
    elif ruta == 'Iezzi':
        
        nombre = 'NOA1348'
        
    return nombre

def n_promedio(info:list):
    
    j = 0
    suma = 0
    
    for i in info:
        
        j += 1
        n = i[1]
        suma += n
        
    promedio = suma/j
    
    return promedio

def n_desviación_estandar(info:list,promedio:float):
    
    j = 0
    sumatoria = 0
    
    for i in info:
        
        j += 1
        n = i[1]
        resta = pow(n - promedio,2)
        sumatoria += resta
        
    des_est = math.sqrt(sumatoria/(j-1))
        
    return des_est    

def graficar_índice_de_refracción_kn (enlace:str, info:list, promedio:float, des_est:float, dic:dict):
    
    Lista_Onda = []
    Lista_n = []
    
    promedio_str = str(promedio)
    des_est_str = str(des_est)
    
    nombre = dic.get(enlace)
    
    for i in info:
       
       Onda = i[0]
       Lista_Onda.append(Onda)
       n = i[1]
       Lista_n.append(n)
           
    plt.plot(Lista_Onda,Lista_n)
    plt.title('Grafica del índice de refracción del material ' + "'" + nombre + "'" + ' en función de su longitud de onda' +
              '\nÍndice de refracción promedio de ' + "'" + nombre + "'" + ": " + promedio_str +
              '\nDesviación estándar del índice de refracción de ' + "'" + nombre + "'" + ': ' + des_est_str)
    
    plt.xlabel('Longitud de Onda')
    plt.ylabel('Índice de refracción')
    
    plt.show()

#graficar_índice_de_refracción_kn('https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/glass/lzos/BF1.yml',crear_list_tupla_onda_y_n('BF1.yml'),n_promedio(crear_list_tupla_onda_y_n('BF1.yml')),n_desviación_estandar(crear_list_tupla_onda_y_n('BF1.yml'),n_promedio(crear_list_tupla_onda_y_n('BF1.yml'))),crear_enlace_material_dic('indices_refraccion.csv'))
#crear_enlace_material_dic('indices_refraccion.csv')
buscar_enlace()
