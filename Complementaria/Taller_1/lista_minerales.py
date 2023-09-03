# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:16:54 2023

@author: elect
"""

from mineral import Mineral

def lista_minerales(ruta:str):
    
    lista_minerales = []
    archivo = open(ruta, 'r')
    lineas = archivo.readlines()
    
    for i in lineas[1:]:
    
        datos = i.strip().split()
        j = 0
        
        while j < len(datos):
            
            if datos[j] == 'NO':
                
                datos[j] = datos[j] + ' '+ datos[j+1]
                datos.pop(j+1)
                j = len(datos) + 1
                
            elif datos[j] == 'METÁLICO/NO':
                
                datos[j] = datos[j] + ' ' + datos[j+1]
                datos.pop(j+1)
                j = len(datos) + 1
                
            else:
                
                j += 1
            
        nombre = datos[0]
        dureza = float(datos[1])
        lustre = datos[5]
        rompimiento_por_fractura = datos[2]
        color = datos[3]
        composición = datos[4]
        sistema_cristalino = datos[7]
        specific_gravity = float(datos[6])
        
        mineral = Mineral(nombre,dureza,lustre,rompimiento_por_fractura,color,composición,sistema_cristalino,specific_gravity)
        lista_minerales.append(mineral)
    
    return lista_minerales

#print(lista_minerales('minerales.txt'))

def cuantos_silicatos_hay(lista_minerales:list):
    
    n_silicatos = 0
    
    for i in lista_minerales:
        
        silicato = i.clasificacion_silicato()
        
        if silicato == True:
            
            n_silicatos += 1
    
    return n_silicatos

#print(cuantos_silicatos_hay(lista_minerales('minerales.txt')))

def densidad_promedio_SI(lista_minerales:list):
    
    n = 0
    sumatoria = 0
    
    for i in lista_minerales:
        
        densidad = i.densidad_SI()
        densidad_float = float(densidad.strip().split()[0])
        sumatoria += densidad_float
        n += 1
        
    densidad_prom = sumatoria/n
    densidad_prom_str = f'{densidad_prom} kg/m^3'
    
    return densidad_prom_str

#densidad_promedio_SI(lista_minerales('minerales.txt'))