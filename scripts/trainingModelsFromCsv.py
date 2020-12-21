
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

#########
# Useful data from converting to csv
dummies = [0,1,2,3,4,42,43,44,45,46,47,48]
   
###########
# prepare sequences
nbOfNote = 49 # taken from tab2csvForTraining
seqLength = 7 
inputsSeq=[[] for i in range(nbOfNote)]
outputsSeq=[[] for i in range(nbOfNote)]
csvDir = "../outputresources/csvs/"

for i in range(nbOfNote):
    if not i in dummies:
        f = csvDir + str(i) + ".csv"
        df = pd.read_csv(f)
        outputsSeq.append(list(df['output']))
        df.drop('output', axis=1)
        inputsSeq.append(df.values.tolist())

allNotePosSet = set(inputsSeq)
# Number of different (notename, position used to play it) exists
nbOfTuple = len(allNotePosSet)
print(nbOfTuple)

###########
# Preparing model list
# a list with a length of nbOfNote but the dummy ones will not be used
# models = [None for i in range(nbOfNote)]
# for i in range(nbOfNote):
#     if not i in dummies:
#         model = Sequential()
#         model.add(GRU(neurons, implementation=impl, recurrent_dropout=drop))
#         models[i] = model
