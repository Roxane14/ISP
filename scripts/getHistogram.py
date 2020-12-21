# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:09:31 2020

@author: Paul Bruneau
"""

import os
import numpy as np
#from music21 import tablature
from preprocess import extractTuples
from FretToPitch import getAssociatedNote

folder = "D:\\Documents\\Ensim\\S9b\\ISP\\git_isp\\ISP\\Tab\\onlytab\\"

files = os.listdir(folder)

histogram = np.zeros(49, dtype=int)

list_notes = []

for fret in range(0,23+1):
    list_notes.append(str(getAssociatedNote(1,fret)))
for fret in range(0,24+1):
    list_notes.append(str(getAssociatedNote(6,fret)))
    

for name_file in files:
    print(name_file)
    
    file = open(folder + name_file)
    tuples = extractTuples(file)
    
    for tpl in tuples:
        note = str(getAssociatedNote(tpl[0], tpl[1]))
        histogram[list_notes.index(note)] += 1
        

for i in range(len(histogram)):
    print(histogram[i])
