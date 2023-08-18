# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 19:35:06 2023

@author: elect
"""

import prueba as p

def cargar_yml():
    
    archivo = input("aaaaa: " )
    a2 = p.crear_list_tupla_onda_y_n(archivo)
    
    return a2

def mostrar_menu():
    
    print("\n")
    print("1. Cargar la vaina esa.")
    
def iniciar():
    
    continuar = True
    
    while continuar:
        mostrar_menu()
        opcion_sec = int(input("selecciona el 1. "))
        if opcion_sec == 1:
            archivo = cargar_yml()
        elif opcion_sec != 1:
            continuar = False
            
iniciar()
