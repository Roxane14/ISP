
import glob
import time
import numpy as np
from music21 import tablature, scale
from FretToPitch import getAssociatedNote
from PitchToFrets import getPossibleTuples
from preprocess import extractTuples
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


import pandas as pd

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from keras.models import model_from_json


# Plot the training and validation loss + accuracy
def plot_training(history):
    outdir = "../outputresources/trainingrnn2/"
    matplotlib.use('Agg')

    #Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig(outdir+ 'fine_tuning_accuracy.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig(outdir+'fine_tuning_loss.pdf')
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


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
csvDir = "../outputresources/csvs_no_outliers2DATAFIX200MIN/"
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
            # If only 1 position is mainly played (as this is based now on csvs2 which
            # removes position that appears less than 10 times in all dataset) 
            # then we removed corresponding data and we consider it as a dummy note where 
            # the main position will be attributed for each position to predict
            if nbPossiblePosPerNote[i] <= 1:
                dummies.append(i)
            print(f"nb different output for {i} is {nbPossiblePosPerNote[i]}")
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
with open("../outputresources/globseqIdsDATAFIX200MIN.txt", 'r') as f:
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
# for i in range(nbOfNote):
#     if not i in dummies:
        
# METRICS = [
#     keras.metrics.TruePositives(name='tp'),
#     keras.metrics.FalsePositives(name='fp'),
#     keras.metrics.TrueNegatives(name='tn'),
#     keras.metrics.FalseNegatives(name='fn'), 
#     keras.metrics.BinaryAccuracy(name='accuracy')]

##########
# Training models
bsize = 64*4
epoch_it = 10*2

for i in range(nbOfNote):
    if not i in dummies:
        ########
        # creation of training/validation data
        nbInput = inputsSeq[i].shape[0]
        indices = np.random.permutation(nbInput)
        splitIndiceMax = int(val_split * nbInput)
        training_idx, test_idx = indices[:splitIndiceMax], indices[splitIndiceMax:]
        training, test = inputsSeq[i][training_idx,:], inputsSeq[i][test_idx,:]
        # training_output, test_output = outputsSeq[i][training_idx], outputsSeq[i][test_idx]
        # training, validating, training_output, validating_output = train_test_split(inputsSeq[i], outputsSeq[i], test_size=0.05)
        training = inputsSeq[i]
        training_output = outputsSeq[i]
        test_x = inputsSeq_test[i]
        
        # categorize output values
        y_train = to_categorical(training_output, nbPossiblePosPerNote[i])
        test_y = to_categorical(outputsSeq_test[i], nbPossiblePosPerNote[i])
        # y_val = to_categorical(test_output, nbPossiblePosPerNote[i])
        
        # training reshape
        print(f"trainingshape {i}")
        print(training.shape)
        training = np.reshape(training, (len(training), seqLength, 1))
        print(training.shape)
        # testing reshape
        print(f"testingshape {i}")
        print(test_x.shape)
        test_x = np.reshape(test_x, (len(test_x), seqLength, 1))
        print(test_x.shape)

        if training.shape[0] == 0:
            print(training)

        ##########
        # computing weights
        
        # for each possipble pos calculate number of occurences in output
        nb_occur_per_class = Counter(training_output)
        total = sum(nb_occur_per_class.values())
        print("nb occurences")
        print(nb_occur_per_class)
        # then compute weights 
        weights={}
        for j in range(nbPossiblePosPerNote[i]):
            nb_occ = nb_occur_per_class[j]
            w = (1 / nb_occ)*(total)
            weights[j] = w
        print(f"{weights=}")

        #########
        # creation of model
        model = None
        model = Sequential()
        # model.add(Embedding(numTuple + 1, 32, input_length=seqLength))
        model.add(GRU(8 , input_shape=(training.shape[1], training.shape[2]), return_sequences=True, implementation=2))
        model.add(GRU(8 , return_sequences=False, implementation=2))
        model.add(Dense(nbPossiblePosPerNote[i]))
        model.add(Activation('softmax'))
        # learning_rate = 0.01
        # optimizer = SGD(lr=learning_rate, momentum=0.95)
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

        print(model.summary())

        ##########
        # trianing
        history = model.fit(training,  y_train, batch_size=bsize, epochs=epoch_it,class_weight=weights)#,validation_data=(validating, validating_output) validation_data=(test_x, test_y))
        ###########
        # Plot evolution of accuracy and lost
        # plot_training(history)

        ##########
        # evaluate model and show confusion matrices
        
        score = model.evaluate(test_x, test_y,
                                    batch_size=bsize)
        print(f'\n\nFor {i}: Test ACC={score}')

        Y_pred = model.predict(test_x)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        conf_mat = confusion_matrix(outputsSeq_test[i], y_pred)
        print(conf_mat)
        print('Classification Report')
        print(classification_report(outputsSeq_test[i], y_pred))

        # confmats = multilabel_confusion_matrix(outputsSeq_test[i], y_pred)
        # weights = [None for i in range(nbPossiblePosPerNote[i])]
        # for i in range(nbPossiblePosPerNote[i]):
        #     weights[i] = 


        # plt.figure()
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # # plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
        # plot_roc("Test Baseline", outputsSeq_test[i], y_pred, color=colors[0], linestyle='--')
        # plt.legend(loc='lower right')
        # plt.savefig("../outputresources/ROCplotNN.png")




        # title = f"note {i}"
        # # print(f'For {i}: Test ACC={score}')
        # print("AAA", np.unique(outputsSeq_test[i]))
        # disp = plot_confusion_matrix(model, inputsSeq_test[i], outputsSeq_test[i],
        #                          #display_labels=[],
        #                          cmap=plt.cm.Blues,
        #                          normalize=None)
        # disp.ax_.set_title(title)    
        # plt.savefig("../outputresources/confmatrnn1/" + title+".png")
        # print(title)
        # print(disp.confusion_matrix)

        models[i] = model

        #############
        # SAVING model
        
        # Saving model and weights
        model.save(f'../outputresources/modelsLSTMlr001epoch20bs16_200MIN_fixunbalanceddata2/model{i}')
        # model_json = model.to_json()
        # with open(f'../outputresources/modelsLSTMlr001epoch20bs16_200MIN_fixunbalanceddata/model{i}.json', 'w') as json_file:
        #         json_file.write(model_json)
        # weights_file = f"../outputresources/modelsLSTMlr001epoch20bs16_200MIN_fixunbalanceddata/weights{i}.hdf5" #f"../outputresources/modelsSVMs_fromCsvs_no_outliers2/{i}.joblib"
        # model.save_weights(weights_file,overwrite=True)



#############
# Results



# for i in range(nbOfNote):
#     if not i in dummies:
#         test_x = inputsSeq_test[i]
#         test_y = to_categorical(outputsSeq_test[i], nbPossiblePosPerNote[i])
#         score = models[i].evaluate(test_x, test_y,
#                                     batch_size=bsize)
#         print()
#         print(f'For {i}: Test ACC={score}')


