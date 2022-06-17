# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:04:17 2022

@author: pc
"""

#%% Model Deployment
import os
import re
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

#1) trained model >> h5
#2) tokenizer >> json
#3) mms/ohe .>> pickle file

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')

# to load trained model
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))
loaded_model.summary()
# to load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%% Input Deployment

input_review = 'The movie is so good, the trailer intrigues me to watch.\
                    The movie is funny, I love it so much.'
# input_review = input('PLease give me your review here')

#preprocessing
input_review = re.sub('<.*?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ', input_review).lower().split()

# to import tokenizer_from_json
tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = pad_sequences(np.array(input_review_encoded).T,maxlen=180,
                                     padding='post',
                                     truncating='post')
# convert into array and .T for horizontal data array

outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)

print(loaded_ohe.inverse_transform(outcome))