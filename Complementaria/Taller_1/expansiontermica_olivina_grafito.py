# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:00:10 2023

@author: elect
"""

from expansiontermicamineral import ExpansionTermicaMineral
import lista_minerales as lm
import matplotlib.pyplot as plt
from pathlib import Path

def olivino_y_grafito(lista_minerales):
    
    i = 0
    grafito = 0
    olivino = 0
    
    while i < len(lista_minerales):
        
        mineral = lista_minerales[i]
        nombre = mineral.nombre
        
        if nombre == 'grafito':
            
            grafito = ExpansionTermicaMineral(nombre, mineral.dureza, mineral.lustre, mineral.rompimiento_por_fractura, mineral.color, mineral.composicion, mineral.sistema_cristalino, mineral.gravedad_especifica, 'graphite_mceligot_2016.csv',[],[])
            i += 1
            
            if olivino != 0:
                
                i = len(lista_minerales)
            
        elif nombre == 'olivino':
            
            olivino = ExpansionTermicaMineral(nombre, mineral.dureza, mineral.lustre, mineral.rompimiento_por_fractura, mineral.color, mineral.composicion, mineral.sistema_cristalino, mineral.gravedad_especifica, 'olivine_angel_2017.csv',[],[])
            i += 1
            
            if grafito != 0:
                
                i = len(lista_minerales)
            
        else:
            
            i += 1
            
    tupla = (grafito,olivino)   
            
    return tupla
            
def graficas_y_coeficientes(tupla:tuple):
    
    ruta = str(Path.cwd())
    
    gra = tupla[0]
    oli = tupla[1]
    
    tupla_gra = gra.calcular_y_visualizar_coeficiente_expansion_termica()
    plt.savefig(ruta + '\\' + 'grafito.png', bbox_inches='tight')
    plt.clf()
    
    tupla_oli = oli.calcular_y_visualizar_coeficiente_expansion_termica()
    plt.savefig(ruta + '\\' + 'olivino.png', bbox_inches='tight')
    plt.clf()
    
    tupla_gra_oli = (tupla_gra,tupla_oli)
    
    return tupla_gra_oli
    
#print(graficas_y_coeficientes(olivino_y_grafito(lm.lista_minerales('minerales.txt'))))