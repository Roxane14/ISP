"""
Objective of this file is to put the basic structure of the training of models

TODO:
- utiliser indications comme "Track 1: Lead Guitar" pour savoir ou sont les solos

"""
import glob
import time
import numpy as np
from music21 import tablature, scale
from FretToPitch import getAssociatedNote
from PitchToFrets import getPossibleTuples
from preprocess import extractTuples

def getAllNotes():
    notes = set()
    for fret in range(0,25):
        for string in range(1,7):
            notes.add( getAssociatedNote(string, fret))
    return list(notes)

# tab vector => noteName
# tab vector => (noteName, position number)
# feed Model associated with noteNamte with sequence of (noteName, position number)
    # note : dont do model for notes that can be played only in one position 


if __name__ == "__main__":
    starttime = time.time()

    ##########
    # Metadata

    # Get all playable notes of guitar ordered by pitch
    allNotes = sorted(getAllNotes(), key=lambda x:x.frequency)
    allNoteNames = [str(n) for n in allNotes]
    possiblePosPerNote = [getPossibleTuples(n) for n in allNoteNames]
    # In order to normalize values between 0 and 1
    maxNbOfPossiblePos = 0
    nbOfNote = len(allNotes)

    def getPosId(n, s, f):
        return possiblePosPerNote[n].index([s,f]) 
    

    # to store notes that have only one position on guitar fretboard
    dummyNotes = [] 
    dummyPos = []
    for i in range(len(possiblePosPerNote)):
        nbOfPos = len(possiblePosPerNote[i])
        maxNbOfPos = max(maxNbOfPos, nbOfPos)
        if (nbOfPos == 1):
            dummyNotes.append(allNoteNames[i])
            dummyPos.append(possiblePosPerNote[i][0])

    # Converter between int and notename / (string, fret)
    id2notename = {i:allNoteNames[i] for i in range(len(allNoteNames))}
    notename2id = {v:u for (u,v) in id2notename.items()}

    ###########
    # load data
    seqLength = 5
    seqsPerNote = []
    trainDir = "../Tab/onlytab/"
    testDir = "" #TODO
    inputsSeq=[]
    outputsPos=[]
    for f in glob.glob(trainDir+"tab_18.tab"): #TODO: change to just *.tab
        with open(f) as ftab:
            fretstringList = extractTuples(f)
            noteNameList = [getAssociatedNote(t, s) for [t,s] in fretstringList]
            noteIdList = [notename2id[i] for i in noteNameList]
            nbNote = len(fretstringList)
            for i in range(seqLength, nbNote-seqLength-1):
                seqNote = noteIdList[i:i+seqLength]
                outputString, outputFret = fretstringList[i+seqLength+1]
                outputPos = getPosId(noteIdList[i+seqLength+1], outputString, outputFret)
                outputNote = noteIdList[i+seqLength+1]
                l.append()

    # Maybe better to transform every input as integer with id => note, pos for every possible combination 
    # load data 
        # take all sequence of tab vector and put it in a list
    # assigner id pour chaque combinaison 
       
    ###########
    # prepare sequences
    # []
        # [sequences, output]
            # sequences = [...,[fret, string],...]
            # output = [fret, string]
    # []
        # [sequences, output]
            # sequences = [...,[noteID, posID],...]
            # output = [noteId, posID]

    # seqs[Note numbers, sequence id, ] 

    seqs = [] 

    ###########
    # train

    print(f'Terminated in {time.time() - starttime} !')
