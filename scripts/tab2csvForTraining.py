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
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


from joblib import dump, load

def getAllNotes():
    notes = set()
    for fret in range(0,25):
        for string in range(1,7):
            notes.add( getAssociatedNote(string, fret))
    return list(notes)

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
    if (nbOfPos == 1):
        dummyNotes.append(allNoteNames[i])
        dummyPos.append(possiblePosPerNote[i][0])

# Converter between int and notename / (string, fret)
id2notename = {i:allNoteNames[i] for i in range(len(allNoteNames))}
notename2id = {v:u for (u,v) in id2notename.items()}

###########
# load data

trainDir = "../Tab/onlytab/"
testDir = "" #TODO

print(notename2id)

# we do not differentiate each songs as overlapping sequence betweens 2 different scores 
# will be less present in the data than notes inside the score 
globSeq = []
stringFretList=None
sf2tuple = {}

for f in glob.glob(trainDir+"*.tab"): 
    with open(f) as ftab:
        try:
            stringFretList = extractTuples(ftab)
        except:
            print(f"Error found in {f} while extracting tuples")
            continue
        for i in range(len(stringFretList)):
            [st,fr] = stringFretList[i]
            noteName = str(getAssociatedNote(st, fr))
            if noteName in notename2id.keys():
                noteId = notename2id[noteName]
                posId = getPosId(noteId, st, fr)
                globSeq.append((noteId, posId))
                if not (st,fr) in sf2tuple.keys():
                    sf2tuple[(st,fr)] = (noteId, posId)
            else:
                print(f"Error {noteName} not found in dic, for file {f}")

allNotePosSet = set(globSeq)
tuple2sf = {v:u for (u,v) in sf2tuple.items()}
tuple2id = {t:i for i,t in enumerate(allNotePosSet)}
id2tuple = {v:u for (u,v) in tuple2id.items()}
globSeqIds = [tuple2id[t] for t in globSeq]

id2sf = {i:tuple2sf[t] for (i,t) in id2tuple.items()}
sf2id = {v:u for (u,v) in id2sf.items()}
dump(id2sf, "id2sfcsvs_no_outliers2.joblib")
dump(sf2id, "sf2idcsvs_no_outliers2.joblib")

dump(tuple2sf, "tuple2sfcsvs_no_outliers2.joblib")
dump(sf2tuple, "sf2tuplecsvs_no_outliers2.joblib")
dump(tuple2id, "tuple2idcsvs_no_outliers2.joblib")
dump(id2tuple, "id2tuplecsvs_no_outliers2.joblib")
dump(id2notename, "id2notenamecsvs_no_outliers2.joblib")
dump(notename2id, "notename2idcsvs_no_outliers2.joblib")

dump(possiblePosPerNote, "possiblePosPerNote.joblib")

print(f"{len(allNotePosSet)=}")

with open("../outputresources/globseqIdsDATAFIX200MIN.txt", "w") as f:
    for item in globSeqIds:
        f.write(f"{item} ")

###########
# prepare sequences
seqLength = 7 
inputsSeq=[[] for i in range(nbOfNote)]
outputsSeq=[[] for i in range(nbOfNote)]

for i in range(len(globSeq)-seqLength - 1 ): # last note is part of the answers
    noteToPredictId, posToPredict = globSeq[i+seqLength+1]
    seq = globSeqIds[i:i+seqLength]
    inputsSeq[noteToPredictId].append(seq)
    outputsSeq[noteToPredictId].append(posToPredict)

remove_outlier = True
nbMinPosPlayed = 200

for nID in range(nbOfNote):
    df = pd.DataFrame(np.matrix(inputsSeq[nID]))
    if len(df.index) == len(outputsSeq[nID]):
        df['output'] = outputsSeq[nID]
        if remove_outlier:
            # val_count = df['output'].value_counts()
            # pos_to_rm = val_count[val_count<10].values
            # print(f"{nID} a {pos_to_rm} parceque {val_count} ")
            #  dff[dff[:][0].isin(vc<10)]
            # df = df[df['output'].isin(val_count<10)]
            # df.drop(df.loc[df[0]==0].index, inplace=True)
            print(f"\n\n{nID}")
            print(df['output'].value_counts())
            df['count'] = df.groupby('output')['output'].transform('count')
            df.drop(df.loc[df['count']<=nbMinPosPlayed].index, inplace=True)
            print(df['output'].value_counts())
            del df['count'] 
        df.to_csv(f"../outputresources/csvs_no_outliers2DATAFIX200MIN/{nID}.csv")


###########
# train

print(f'Terminated in {time.time() - starttime} !')
