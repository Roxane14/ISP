# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 22:43:25 2020

@author: Paul Bruneau
"""

file = open('tab_4.tab')
Lines = file.readlines()

count = 0

def getSingleVec(vecset, i):
    return[row[i] for row in vecset]

        
standard_tuning = list("EBGDAE")
tuples = [] #(string, fret)
vecset = list("EBGDAE")

print(standard_tuning)
for line in range(len(Lines)-6):
    count += 1
    vec = []
    for i in range(6):
        vec.append(Lines[line + i])
        vec[i] = vec[i][0]
        if (vec == standard_tuning):
            print("ok!")
            for i in range(line,line+6):
                vecset[i-line] += Lines[i]
                
vecset2 = [list(vecset[0]),
           list(vecset[1]),
           list(vecset[2]),
           list(vecset[3]),
           list(vecset[4]),
           list(vecset[5])]


for i in range(len(vecset2[0])):
    vec = getSingleVec(vecset2, i)
    if (vec.count('-') == 5):
        for element in range(6):
            if (vec[element] != '-'):
                tuples.append([6-element,vec[element]]) 
    