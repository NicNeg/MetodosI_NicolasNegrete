# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 19:17:39 2023

@author: elect
"""

from pathlib import Path
import os
import math
import matplotlib.pyplot as plt

#Funciones base

def buscar_archivo() -> str:
     
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
        
        n = 0
        
        if numero == j:
            
            lista = dic_archivos[j]
            
            while n < len(lista):
                    
                s_metiche = lista[n]
                s_metiche_c = s_metiche[len(s_metiche)-4:len(s_metiche)]
                    
                if s_metiche_c == '.txt':
                        
                    lista.pop(n)
                        
                elif s_metiche_c == '.png':
                        
                    lista.pop(n)
                        
                else:
                        
                    n += 1
            
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

def buscar_enlace(archivo:str) -> str:
    
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

def crear_enlace_material_dict() -> dict:
    
    dict_categorias = {}
    
    ruta1 = Path.cwd() / 'Categorias'
    lista_ruta1 = os.listdir(ruta1)
    
    k = 0
    
    for i in lista_ruta1:
        
        ruta2 = Path.cwd() / 'Categorias' / lista_ruta1[k]
        lista_ruta2 = os.listdir(ruta2)
        
        n = 0
        
        while n < len(lista_ruta2):
            
            s_metiche = lista_ruta2[n]
            s_metiche_c = s_metiche[len(s_metiche)-4:len(s_metiche)]
            
            if s_metiche_c == '.txt':
                
                lista_ruta2.pop(n)
                
            elif s_metiche_c == '.png':
                
                lista_ruta2.pop(n)
                
            else:
                
                n += 1
        
        dict_archivos = {}
        k += 1
        w = 0
        
        for j in lista_ruta2:
            
            nombre_m = lista_ruta2[w]
            nombre_d = len(nombre_m) - 4
            nombre = nombre_m[0:nombre_d]
            dict_archivos[j] = nombre
            dict_categorias[i] = dict_archivos
            w += 1
            
    return dict_categorias

#Función 1.3

def crear_list_tupla_onda_y_n(ruta:str) -> list:
    
    lista = []
    onda = None
    onda2 = None
    continuar = True
    k = 0
    n = None
    
    archivo = open(ruta, 'r', encoding = 'utf-8')
    
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

#Funciones 1.4

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
    
    if len(info) == 1:
        
        des_est = 'No hay'
    
    else:
        
        for i in info:
        
            j += 1
            n = i[1]
            resta = pow(n - promedio,2)
            sumatoria += resta
        
        des_est = math.sqrt(sumatoria/(j-1))
        
    return des_est    

def graficar_índice_de_refracción(archivo:str, info:list, promedio:float, des_est:float, dic:dict):
    
    Lista_Onda = []
    Lista_n = []
    
    j = 0
    
    promedio_str = str(promedio)
    des_est_str = str(des_est)
    lista_tu = list(dic.items())
    
    while j < len(lista_tu):
        
        tupla = lista_tu[j]
        dic_archivos = tupla[1]
        llaves = list(dic_archivos.keys())
        
        k = 0
        
        while k < len(llaves):
        
            if archivo == llaves[k]:
            
                nombre = dic_archivos[archivo]
                j = len(lista_tu)
                k = len(llaves)
            
            else:
            
                k += 1
                
        j += 1
            
    for i in info:
       
       Onda = i[0]
       Lista_Onda.append(Onda)
       n = i[1]
       Lista_n.append(n)
           
    plt.plot(Lista_Onda,Lista_n, color= 'r')   
    plt.scatter(Lista_Onda,Lista_n)
    plt.title('Grafica del índice de refracción del material ' + "'" + nombre + "'" + ' en función de su longitud de onda' +
              '\nÍndice de refracción promedio de ' + "'" + nombre + "'" + ": " + promedio_str +
              '\nDesviación estándar del índice de refracción de ' + "'" + nombre + "'" + ': ' + des_est_str)
    
    plt.xlabel('Longitud de Onda')
    plt.ylabel('Índice de refracción')
    
    plt.show()
    
#Función 1.5

def graficar_todos_los_indices_de_refracción_y_guardarlos(dic:dict):
    
    ruta1 = Path.cwd() / 'Categorias'
    lista_ruta1 = os.listdir(ruta1)
    
    lista_tu = list(dic.items())
    
    for i in range(0,len(lista_ruta1)):
        
        tupla_archivos = lista_tu[i]
        dict_archivos = tupla_archivos[1]
        llaves = list(dict_archivos.keys())
        
        for j in range(0,len(llaves)):
            
            Lista_Onda = []
            Lista_n = []
            ruta2 = Path.cwd() / 'Categorias' / lista_ruta1[i] / llaves[j]
            ruta3 = str(Path.cwd() / 'Categorias' / lista_ruta1[i])
            tuplas = crear_list_tupla_onda_y_n(ruta2)
            nombre = dict_archivos[llaves[j]]
            promedio = n_promedio(tuplas)
            promedio_str = str(promedio)
            des_est_str = str(n_desviación_estandar(tuplas, promedio))
            
            for k in tuplas:
            
                Onda = k[0]
                Lista_Onda.append(Onda)
                n = k[1]
                Lista_n.append(n)
                
            plt.plot(Lista_Onda,Lista_n, color= 'r')   
            plt.scatter(Lista_Onda,Lista_n)
            plt.title('Grafica del índice de refracción del material ' + "'" + nombre + "'" + ' en función de su longitud de onda' +
                      '\nÍndice de refracción promedio de ' + "'" + nombre + "'" + ": " + promedio_str + '\nDesviación estándar del índice de refracción de '+ 
                      "'" + nombre + "'" + ': ' + des_est_str)
                
            plt.xlabel('Longitud de Onda')
            plt.ylabel('Índice de refracción')

            plt.savefig(ruta3 + '\\' + nombre +'.png', bbox_inches='tight')
            plt.clf()

'''archivo = buscar_archivo()
info = crear_list_tupla_onda_y_n(buscar_enlace(archivo))
promedio = n_promedio(info)
des_est = n_desviación_estandar(info, promedio)
dic = crear_enlace_material_dict()'''

#graficar_todos_los_indices_de_refracción_y_guardarlos(dic)

#graficar_índice_de_refracción(archivo,info,promedio,des_est,dic)
#crear_enlace_material_dic('indices_refraccion.csv')
#crear_list_tupla_onda_y_n(buscar_enlace(buscar_archivo()))
#crear_enlace_material_dict()
