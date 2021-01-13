import glob
import time
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from music21 import converter, instrument, note, chord, environment, tempo
from music21 import tablature, scale
from FretToPitch import getAssociatedNote
from PitchToFrets import getPossibleTuples
from preprocess import extractTuples

from sklearn import svm
from joblib import dump, load

from keras.models import model_from_json, load_model
# from tensorflow import keras
import json
"""
Objective of this script is to take models that have been saved after training
in order to:
- import midi file
- convert it again to understandable data for the models
- use models to predict future position
- transform this list of (note, position) to list of tablature vector and display it

"""
nbOfNote = 49

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

def use_rnns(input_file, models_dir= "../outputresources/modelsLSTMlr001epoch20bs16_200MIN_fixunbalanceddata2/"):
    models = [None for i in range(nbOfNote)]
    for i in range(nbOfNote):
        try:
            model = load_model(f'../outputresources/modelsLSTMlr001epoch20bs16_200MIN_fixunbalanceddata2/model{i}')
            # model = model_from_json(models_dir+f"model{i}.json")
            # json_file = open(models_dir+f"model{i}.json", 'r')
            # architecture = json.load(json_file)
            # model = model_from_json(json.dumps(architecture)) 
            # model = model_from_json(architecture)
            # model = model.load_weights(models_dir+f'weights{i}.hdf5')
            models[i] = model
        except Exception as e:
            print("error while loading model",e)
            models[i] = None
            dummies.append(i)
    print(dummies)
    
    # READING Input file
    noteIDs_midi = []
    midi = converter.parse(input_file)
    s = instrument.partitionByInstrument(midi)
    notes_to_parse = s.recurse() 
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            noteIDs_midi.append(notename2id[element.nameWithOctave])

    #########
    # PREDICTING
    # We initialize first seq to lowest note (no fingers needed)
    # seq = [ sf2id[(1,0)] , sf2id[(4,7)], sf2id[(6,5)], sf2id[(5,8)], sf2id[(5,5)], sf2id[(4,7)],sf2id[(3,7)]]
    seq = [ sf2id[(1,0)] for i in range(seqLength)]
    output_sfs = [] #list of string frets predicted
    for i in range(len(noteIDs_midi)):
        nID = noteIDs_midi[i]
        # If there is no model attached or solution is trivial
        if nID in dummies:
            # "predict" the usual assiciated pos
            # TODO: dictionnary for each dummy note id => gives usual position
            predicted_sf = tuple(possiblePosPerNote[nID][0])
        else:
            seq_shaped = np.reshape(seq, (1, seqLength, 1))
            predicted_pos_probs = models[nID].predict(seq_shaped)
            predicted_pos = np.argmax(predicted_pos_probs)
            print(predicted_pos)
            predicted_sf = tuple(tuple2sf[(nID, predicted_pos)])
        output_sfs.append(predicted_sf)
        # then we have to update sliding window
        # print(i)
        # print(nID)
        # print(possiblePosPerNote[nID][0])
        # print(predicted_sf)
        # print(sf2id[predicted_sf])

        seq = seq[1:]+[sf2id[predicted_sf]]
    return output_sfs

def use_svms(input_file, models_dir):
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
    ################
    # READING Input file
    noteIDs_midi = []
    midi = converter.parse(input_file)
    s = instrument.partitionByInstrument(midi)
    notes_to_parse = s.recurse() 
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            noteIDs_midi.append(notename2id[element.nameWithOctave])

    #########
    # PREDICTING
    # We initialize first seq to lowest note (no fingers needed)
    # seq = [ sf2id[(1,0)] , sf2id[(4,7)], sf2id[(6,5)], sf2id[(5,8)], sf2id[(5,5)], sf2id[(4,7)],sf2id[(3,7)]]
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
        # print(i)
        # print(nID)
        # print(possiblePosPerNote[nID][0])
        # print(predicted_sf)
        # print(sf2id[predicted_sf])

        seq = seq[1:]+[sf2id[predicted_sf]]
    return output_sfs


if __name__=='__main__':
    models_dir = "../outputresources/modelsLSTMlr001epoch20bs16_200MIN_fixunbalanceddata3/"#modelsSVMs_fromCsvs_no_outliersDATAFIX2/" #modelsLSTMlr001epoch20bs16_200MIN_fixunbalanceddata/"
    input_file = "../Tab/testing/ledzepSolo.mid"
    translation = use_rnns(input_file, models_dir)
    print(translation)