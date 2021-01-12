# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:47:42 2020

@author: Roxy8
"""

from FretToPitch import getAssociatedNote

def getPossibleTuples(note):
    tuples = []
    for fret in range(0,25):
        for string in range(1,7):
            if str(getAssociatedNote(string, fret)) == note:
                tuples.append([string, fret])
    return tuples


if __name__ == "__main__":
    print("As a test, G4 --> "+str(getPossibleTuples("G2")))