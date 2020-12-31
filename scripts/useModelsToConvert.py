import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from music21 import converter, instrument, note, chord, environment, tempo
from music21 import tablature, scale
from FretToPitch import getAssociatedNote
from PitchToFrets import getPossibleTuples
from preprocess import extractTuples

from sklearn import svm
from joblib import dump, load

"""
Objective of this script is to take models that have been saved after training
in order to:
- import midi file
- convert it again to understandable data for the models
- use models to predict future position
- transform this list of (note, position) to list of tablature vector and display it

"""
nbOfNote = 49
models_dir = "../outputresources/modelsSVMs_fromCsvs_no_outliers2/"
input_file = "../Tab/testing/ledzepSolo.mid"
dummies=[]
seqLength = 7

###############
# get metadata
tuple2sf = load("tuple2sfcsvs_no_outliers2.joblib")
sf2tuple = load("sf2tuplecsvs_no_outliers2.joblib")
# dump(tuple2id, "tuple2idcsvs_no_outliers2.joblib")
# dump(id2tuple, "id2tuplecsvs_no_outliers2.joblib")
id2notename = load("id2notenamecsvs_no_outliers2.joblib")
notename2id = load("notename2idcsvs_no_outliers2.joblib")

id2sf = load("id2sfcsvs_no_outliers2.joblib")
sf2id = load("sf2idcsvs_no_outliers2.joblib")

possiblePosPerNote = load("possiblePosPerNote.joblib")

##########
# LOAD models
models = [None for i in range(nbOfNote)]
for i in range(nbOfNote):
    try:
        models[i] = load(f"{models_dir}{i}.joblib")
    except Exception as e:
        print(e)
        models[i] = None
        dummies.append(i)
print(dummies)

################
# READING Input file
noteIDs_midi = []
midi = converter.parse(input_file)
s = instrument.partitionByInstrument(midi)
notes_to_parse = s.recurse() 
for element in notes_to_parse:
    if isinstance(element, note.Note):
        noteIDs_midi.append(notename2id[element.nameWithOctave])

print(noteIDs_midi)

#########
# PREDICTING
# We initialize first seq to lowest note (no fingers needed)
seq = [ sf2id[(1,0)] for i in range(seqLength)]
output_sfs = [] #list of string frets predicted
for i in range(len(noteIDs_midi)):
    print()
    nID = noteIDs_midi[i]
    # If there is no model attached or solution is trivial
    if nID in dummies:
        # "predict" the usual assiciated pos
        # TODO: dictionnary for each dummy note id => gives usual position
        predicted_sf = tuple(possiblePosPerNote[nID][0])
    else:
        print(f"we predict with model {nID}")
        predicted_pos = models[nID].predict(np.array(seq).reshape(1,-1))
        predicted_sf = tuple(tuple2sf[(nID, predicted_pos[0])])
    output_sfs.append(predicted_sf)
    # then we have to update sliding window
    print(i)
    print(nID)
    print(possiblePosPerNote[nID][0])
    print(predicted_sf)
    print(sf2id[predicted_sf])

    print(seq)
    seq = seq[1:]+[sf2id[predicted_sf]]
    print(seq)


print(output_sfs)


