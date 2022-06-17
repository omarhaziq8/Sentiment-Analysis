# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:17:55 2022

@author: pc
"""

import pandas as pd 
import numpy as np 
import os 
import pickle
import json
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Bidirectional,Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#%% Statics

CSV_URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')

#%% EDA ->>> Data Loading

df = pd.read_csv(CSV_URL)
df_copy = df.copy() # to act as copy to avoid long time loading/backup
#%% Data Inspection 

df.head(10)
df.info()

df.duplicated().sum()
df[df.duplicated()]

#%% Data Cleaning
# remove the <br><br HTMl tags 

df = df.drop_duplicates()

#remove HTML tags
review = df['review'].values # Features: x
sentiment = df['sentiment'].values # sentiment: y
# put .values to make sure the shape of dataset is linked are the same
import re 
for index,rev in enumerate(review):
    # remove html tags 
    # ? dont be greedy
    # * zero or more occurences 
    # any character except new line (/n)
    
    review[index] = re.sub('<.*?>',' ',rev)
    
    # convert into lower case
    # remove numbers 
    # ^ means NOT 
    review[index] = re.sub('[^a-zA-Z]',' ', rev).lower().split()


#%% Features Selection
# Nothing to select
#%% Preprocessing (TOKENIZER)
# To convert all the vocab into number within 10000 words
vocab_size = 10000
oov_token = 'OOV'
max_len = 180


tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review)# to learn all the words
word_index = tokenizer.word_index
# print(word_index) to show the words_num dictionary

train_sequences = tokenizer.texts_to_sequences(review) # to convert into numbers

#%% (PADDING)

length_of_review = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_review)) # to get the number of max length for padding


padded_review = pad_sequences(train_sequences,maxlen=max_len,padding='post',
              truncating='post') # refer to x variable

#%% One Hot Encoding for the Target (Classification)


ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

#%% Train Test Split 


x_train,x_test,y_train,y_test = train_test_split(padded_review,sentiment,
                                                 test_size=0.3,
                                                 random_state=123)


x_train = np.expand_dims(x_train,axis=-1) # to reshape since the x train is in 2D
x_test = np.expand_dims(x_test,axis=-1)

#%% Model development



embedding_dim = 64 

model = Sequential()
model.add(Input(shape=(np.shape(x_train)[1]))) # Input shape=(180,1)np.shape(x_train)[1:]
model.add(Embedding(vocab_size, embedding_dim)) # embedding size doesnt accept array (180,1)
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
# model.add(LSTM(128,return_sequences=(True)))
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,'softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics='acc')

hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,
          batch_size=128)

plot_model(model,show_layer_names=(True),show_shapes=(True))
#%%  Plot Visualisation

hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()

# from the graphs plot, shows the graph is underfitting with acc 0.54

#%% Model evaluation 

y_true = y_test 
y_pred = model.predict(x_test)

y_true = np.argmax(y_true,axis=1) # to convert 0/1 
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))


#%% Model saving

#H5 model save
model.save(MODEL_SAVE_PATH)

#Initialise token_json
token_json = tokenizer.to_json()

#to save tokenizer as dictionary
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

# To save ohe
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)


#%% Discussion

# The model is actually not learning so well after adjusting layer with only 0.54
# The graphs shows the model is underfitting to predict the outcome
# So, tu tune the model, we use bidirectional and embedded layer to fit the model
# after training, the results show 84% accuracy, recall 79%, f1 score 83% respectively
# However, after plotting, the grapsh shows overfitting on 2nd epoch
# To overcome this, early stopping can be introduced to prevent it
# we can increase dropout rate to control overfitting
# other than that, can use other DL architecture like BERT model,transformer
# GPT3 model may help to improve

























