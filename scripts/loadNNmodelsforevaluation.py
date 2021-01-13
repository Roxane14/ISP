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

from keras.models import model_from_json

"""
Objective of this script is to take models that have been saved after training
in order to:
- import midi file
- convert it again to understandable data for the models
- use models to predict future position
- transform this list of (note, position) to list of tablature vector and display it

"""
nbOfNote = 49
models_dir = "../outputresources/modelsLSTMlr001epoch20bs16_200MIN/"
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
        model = model_from_json(models_dir+f"model{i}.json")
        model = model.load_weights(models_dir+f'weights{i}.hdf5')
        models[i] = model
    except Exception as e:
        print(e)
        models[i] = None
        dummies.append(i)
print(dummies)

#######
# special for this
i = 12
model = models[i]

########
# load input and output sequences
f = csvDir + str(i) + ".csv"
df = pd.read_csv(f, index_col=0)
# outputsSeq.append(list(df['output']))
# df.drop('output', axis=1)
# inputsSeq.append(df.values.tolist())
outputs = df['output'].values
df = df.drop('output', axis=1)
inputs = df.values
nbInput = inputs.shape[0]


# pb is that sometimes outputs that are appearing less are put
# in 
# solution : séparer pour chaque categorie les indices en train/test
# puis les regrouper pour selectionner dans les listes
training_idx = np.array([])
test_idx = np.array([])

for j in range(3):
    those_indices = np.random.permutation([k for k, x in enumerate(outputs) if x == j])
    if (len(those_indices) <=2):
        print(f'not enough case of posiiton nb {j} in dataset for {i}')
    splitIndiceMax = int((1-test_split) * len(those_indices))     
    training_idx = np.append(training_idx, those_indices[:splitIndiceMax])
    test_idx = np.append(test_idx, those_indices[splitIndiceMax:])

inputsSeq_test = inputs[test_idx.astype(int),:]
outputsSeq_test = outputs[test_idx.astype(int)]


######
# Evaluate
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy')]
