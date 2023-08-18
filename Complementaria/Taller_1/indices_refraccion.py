# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 19:17:39 2023

@author: elect
"""

from pathlib import Path
import os
import math
import matplotlib.pyplot as plt
import pandas as pd

def buscar_archivo () -> str:
     
    ruta1 = Path.cwd() / 'Categorias'
    
    lista_ruta1 = os.listdir(ruta1)
    lista_mensaje = []
    
    print('Elija una carpeta: \n')
    
    for i in range(0,len(lista_ruta1)):
        
        mensaje = str(i+1) + '. ' + lista_ruta1[i]
        lista_mensaje.append(mensaje)
        
    print(*lista_mensaje, sep = '\n')
    numero = int(input("\nEscriba su opción: "))
    
    dic_archivos = {}
    
    for i in range(0,len(lista_ruta1)):
        
        ruta2 = Path.cwd() / 'Categorias' / lista_ruta1[i]
        lista_ruta2 = os.listdir(ruta2)
        
        dic_archivos[i+1] = lista_ruta2
        
    print('\nElija un archivo: \n')
    
    j = 1
    lista_mensaje2 = []
    
    while j < 9:
        
        if numero == j:
            
            lista = dic_archivos[j]
            
            for i in range(0,len(lista)):
            
                mensaje2 = str(i+1) + '. ' + lista[i]
                lista_mensaje2.append(mensaje2)
                
            j = 9
            
        else:
            
            j += 1
            
    print(*lista_mensaje2, sep = '\n')
    archivo_n = int(input("\nEscriba su opción: "))
    archivo = lista[archivo_n-1]
    
    return archivo

def buscar_enlace (archivo:str) -> str:
    
    ruta1 = Path.cwd() / 'Categorias'
    
    lista_ruta1 = os.listdir(ruta1)
    continuar = True
    
    while continuar:
        
        i = 0
        
        while i < len(lista_ruta1):
            
            carpeta = lista_ruta1[i]
            ruta2 = Path.cwd() / 'Categorias' / carpeta
            lista_ruta2 = os.listdir(ruta2)
            
            j = 0
            
            while j < len(lista_ruta2):
                
                if lista_ruta2[j] == archivo:
                    
                    ruta = Path.cwd() / 'Categorias' / carpeta / archivo
                    continuar = False
                    i = len(lista_ruta1) + 1
                    j = len(lista_ruta2) + 1
                    
                else:
                    
                    j += 1
                    
            i += 1
        
    return ruta

def crear_enlace_material_dict () -> dict:
    
    dict_categorias = {}
    
    ruta1 = Path.cwd() / 'Categorias'
    lista_ruta1 = os.listdir(ruta1)
    
    for i in lista_ruta1:
        
        dict_categorias[i]
        ruta2 = Path.cwd() / 'Categorias' / dict_categorias[i]
        lista_ruta2 = os.listdir(ruta2)
        dict_archivos = {}
        
        for j in lista_ruta2:
            
            dict_categorias[i] = dict_archivos[j]
            dict_archivos[j] = 
    
    
        
    return dict_categorias

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
        
    print(lista)
    return lista

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

def graficar_índice_de_refracción_kapton_NOA138 (enlace:str, info:list, promedio:float, des_est:float, dic:dict):
    
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
crear_list_tupla_onda_y_n(buscar_enlace(buscar_archivo()))

