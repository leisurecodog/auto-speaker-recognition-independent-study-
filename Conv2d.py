import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils import to_categorical
from keras.layers.core import Activation,Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import *
from keras.callbacks import *
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import sequence


Batch_Size = 32

#############################################
def D_open1(dirs,i):
    # online data open format
    print('test')
    '''
    count=0
    for name in dirs:
        nn = os.listdir(DIR+'/'+name)
        for NN in nn:
            nnn = os.listdir(DIR+'/'+name+'/'+NN)
            for NNN in nnn:
                count += 1
                Data,sr = librosa.load(DIR+'/'+name+'/'+NN+'/'+NNN,sr=None)
                Datas.append(Data)
                if(MAX_LEN<len(Data)):
                    MAX_LEN=len(Data)
                Y.append(int(i))
        i += 1
        if i == 45:
            break
'''
############################################
def D_open2(dirs):
    # 中研院 data open format
    MAX_LEN = 0
    Datas = []
    Y = []
    i = 0
    for name in dirs:
        nn = os.listdir(DIR+'/'+name)
        for NN in nn:
            Data,sr = librosa.load(DIR+'/'+name+'/'+NN,sr=None)
            Datas.append(Data)
            if(MAX_LEN<len(Data)):
                MAX_LEN=len(Data)
            Y.append(int(i))
        i += 1
        if i == 500:#選擇loading多少個人
            break
    Datas = sequence.pad_sequences(Datas, maxlen=MAX_LEN,dtype='float32')
    X = []
    for xx in Datas:
        tmp = librosa.feature.mfcc(y=xx,sr=sr,n_mfcc=40)
        X.append(tmp)
    Y = np.asarray(Y)
    X = np.asarray(X)
    X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
    return X , Y, i
###########################################

def create_model(X_train,X_test,Y_train,Y_test,i):

    model=Sequential()

    model.add(Conv2D(64,kernel_size=(2,2),input_shape=(X.shape[1],X.shape[2],1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(32,kernel_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,kernel_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,kernel_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(32,activation='relu'))
    # Add output layer
    model.add(Dense(i,activation='softmax'))

    model.compile(loss='categorical_crossentropy'
                  ,optimizer='adam',metrics=["accuracy"])

    # Show model information
    # model.summary()
    ES=EarlyStopping(monitor='val_acc',patience=1000,mode='max')
    model.fit(X_train,Y_train,batch_size=Batch_Size,
          verbose=0,epochs=660)

    score, acc = model.evaluate(X_test, Y_test, batch_size=Batch_Size)
    return acc

###########################################
def normal_train(X,Y,i):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,shuffle=True)
    acc = create_model(X_train,X_test,Y_train,Y_test,i)
    print('acc: {0}%'.format(acc*100))
    

###########################################
def cross_val_train(X,Y,i):
    kfold = StratifiedKFold(n_splits=10,shuffle=False)
    for train,test in kfold.split(X,Y):
        X_train = X[train]
        X_test = X[test]
        Y_train = to_categorical(Y[train])
        Y_test = to_categorical(Y[test])
        acc = create_model(X_train,X_test,Y_train,Y_test,i)
        print('acc: {0}%'.format(acc*100))

if __name__ == '__main__':
    DIR = 'sec' #target folder
    #open files
    dirs = os.listdir(DIR)
    X = []
    Y = []
    i = 0
    X ,Y ,i = D_open2(dirs)
    cross_val_train(X,Y,i)
