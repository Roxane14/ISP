# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:46:34 2020

@author: Roxy8
"""
from music21 import tablature

def getAssociatedNote(stri, fr):
    note = tablature.FretNote(string=stri, fret=fr)
    gfb = tablature.GuitarFretBoard(fretNotes=[note])
    pitches = gfb.getPitches()
    nb = (6-stri)%6
    pitch = pitches[nb]
    return pitch

if __name__ == "__main__":
    # Testing the function with a note
    print("As a test, (12, 3) --> "+str(getAssociatedNote(12, 3)))