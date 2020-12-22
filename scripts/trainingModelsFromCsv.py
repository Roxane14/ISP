
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
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

#########
# Useful data from converting to csv
dummies = [0,1,2,3,4,42,43,44,45,46,47,48]
test_split = 0.2
val_split = 0.2

###########
# prepare sequences
nbOfNote = 49 # taken from tab2csvForTraining
seqLength = 7 
inputsSeq=[None for i in range(nbOfNote)]
outputsSeq=[None for i in range(nbOfNote)]
inputsSeq_test=[None for i in range(nbOfNote)]
outputsSeq_test=[None for i in range(nbOfNote)]
csvDir = "../outputresources/csvs/"
nbPossiblePosPerNote = [None for i in range (nbOfNote)]

# load input and output sequences
for i in range(nbOfNote):
    if not i in dummies:
        f = csvDir + str(i) + ".csv"
        df = pd.read_csv(f, index_col=0)
        # outputsSeq.append(list(df['output']))
        # df.drop('output', axis=1)
        # inputsSeq.append(df.values.tolist())
        outputs = df['output'].values
        df = df.drop('output', axis=1)
        inputs = df.values
        nbInput = inputs.shape[0]
        if not outputs is None:
            nbPossiblePosPerNote[i] = len(set(outputs))
        else:
            print(f"No positions for {i}")
            nbPossiblePosPerNote[i] = 0
        
        # pb is that sometimes outputs that are appearing less are put
        # in 
        # solution : séparer pour chaque categorie les indices en train/test
        # puis les regrouper pour selectionner dans les listes
        training_idx = np.array([])
        test_idx = np.array([])

        for j in range(nbPossiblePosPerNote[i]):
            those_indices = np.random.permutation([k for k, x in enumerate(outputs) if x == j])
            if (len(those_indices) <=2):
                print(f'not enough case of posiiton nb {j} in dataset for {i}')
            splitIndiceMax = int((1-test_split) * len(those_indices))     
            training_idx = np.append(training_idx, those_indices[:splitIndiceMax])
            test_idx = np.append(test_idx, those_indices[splitIndiceMax:])

        inputsSeq[i] = inputs[training_idx.astype(int),:]
        outputsSeq[i] = outputs[training_idx.astype(int)]
        inputsSeq_test[i] = inputs[test_idx.astype(int),:]
        outputsSeq_test[i] = outputs[test_idx.astype(int)]


# Show part of train and test for each note
for i in range(nbOfNote):
    if not i in dummies:
        print(f"For {i} nbtrain = {inputsSeq[i].shape[0]}, nbtest = {inputsSeq_test[i].shape[0]}")
    else:
        print("Dummy note")

# Load full sequence of all songs
globseqIds = []
with open("../outputresources/globseqIds.txt", 'r') as f:
    lines = f.readlines()
    for l in lines:
        ids = [int(x) for x in l.split()]
        globseqIds = globseqIds + ids

allNotePosSet = set(globseqIds)
numTuple = len(allNotePosSet) #123
      

###########
# Preparing model list
# a list with a length of nbOfNote but the dummy ones will not be used
models = [None for i in range(nbOfNote)]
for i in range(nbOfNote):
    if not i in dummies:
        model = Sequential()
        model.add(Embedding(numTuple + 1, 32, input_length=seqLength))
        model.add(GRU(4))
        model.add(Dense(nbPossiblePosPerNote[i]))
        model.add(Activation('softmax'))
        # learning_rate = 0.01
        # optimizer = SGD(lr=learning_rate, momentum=0.95)
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer="adam")
        models[i] = model

##########
# Training models
bsize = 32
epoch_it = 10

for i in range(nbOfNote):
    if not i in dummies:
        # creation of validation data
        nbInput = inputsSeq[i].shape[0]
        indices = np.random.permutation(nbInput)
        splitIndiceMax = int(val_split * nbInput)
        training_idx, test_idx = indices[:splitIndiceMax], indices[splitIndiceMax:]
        training, test = inputsSeq[i][training_idx,:], inputsSeq[i][test_idx,:]
        training_output, test_output = outputsSeq[i][training_idx], outputsSeq[i][test_idx]
        # categorize output values
        y_train = to_categorical(training_output, nbPossiblePosPerNote[i])
        y_val = to_categorical(test_output, nbPossiblePosPerNote[i])
        # training
        models[i].fit(training,  y_train,
                        batch_size=bsize,
                        epochs=epoch_it,
                        validation_data=(test, y_val))


#############
# Results

for i in range(nbOfNote):
    if not i in dummies:
        test_x = inputsSeq_test[i]
        test_y = to_categorical(outputsSeq_test[i], nbPossiblePosPerNote[i])
        score = models[i].evaluate(test_x, test_y,
                                    batch_size=bsize)
        print()
        print(f'For {i}: Test ACC={score}')


