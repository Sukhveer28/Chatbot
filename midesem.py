
import pandas as pd
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import datetime


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [w for w in word_tok if not w in stop_words]
    print(stemmed_words)
    return ' '.join(stemmed_words)


le = LE()

tfv = TfidfVectorizer(min_df=1, stop_words='english')

data = pd.read_csv("/Users/priyanshityagi/Documents/chatbot/BankFAQs.csv") 

questions = data['Question'].values

print(data)

X = []

for question in questions:
    X.append(cleanup(question))

tfv.fit(X)

le.fit(data['Class'])

X = tfv.transform(X)

y = le.transform(data['Class'])

trainx, testx, trainy, testy = tts(X, y, test_size=.3, random_state=42)


model = SVC(kernel='linear')
model.fit(trainx, trainy)


class_=le.inverse_transform(model.predict(X))

accuracy = model.score(testx, testy)

training_data = pd.DataFrame({'Question': questions, 'Predicted_Class': class_})

training_data.to_csv("/Users/priyanshityagi/Documents/chatbot/training_data.csv", index=False)

#print("SVC:", model.score(testx, testy))