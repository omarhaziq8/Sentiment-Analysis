<a><img alt='python' src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a><img alt = 'image' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
<a><img alt='tf' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>
<a><img alt='keras' src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"></a>
<a><img alt='numpy' src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"></a>
<a><img alt='pandas' src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"></a>
<a><img alt='sk-learn' src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"></a>


# Sentiment-Analysis LSTM using Embedding Technique
**Description** : Trained with over 60,000 IMDB dataset to categorized positive and negative review

**Algorithm Model** : Deep Learning method->> LSTM, Long Short Term Memory, BIdirectional 

**Preprocessing step** : TOKENIZER, PADDING, ONE HOT ENCODER

**Objective** : To produce outcome with accuracy 80%~85% range of prediction by model trained

**Flowchart Model** :

<img src="Statics/model.png" alt="Girl in a jacket" style="width:500px;height:600px;"> 

### Exploratory Data Analysis (EDA)
1) Data Loading
2) Data Inspection
3) Data Cleaning
4) Features Selection
5) Pre-Processing

**Model evaluation** :

`Classification_report`
`accuracy_score`
`Confusion_Matrix`
`Model_train_test_split`
`json`
`pickle`
`EDA`


**Discussion** :

 🟠The model is actually not learning so well after increasing neurons layer with only 0.54 accuracy
 
 🟠The graphs shows the model is underfitting to predict the outcome
 
 🟠Therefore, in order for model tuning, we use bidirectional and embedded layer to fit the model in order to improve the accuracy
 
 🟠After training, the results show 84% accuracy, recall 79%, f1 score 83% respectively
 
 🟠However, after plotting, the grapsh shows overfitting on 2nd epoch
 
 🟠To overcome this, early stopping can be introduced to prevent it
 
 🟠We can increase dropout rate to control overfitting
 
 🟠Other than that, can use other DL architecture like BERT model,transformer
 
 🟠GPT3 model may help to improve


**Dataset** :

![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

[Datasets](https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv)


**Enjoy!** 🚀




