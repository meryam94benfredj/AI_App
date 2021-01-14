from flask import Flask, render_template,flash,request,url_for,redirect, session
import numpy as np
import re
import os
import re # Regular expressions
import nltk # Natural language tool kit
from nltk.corpus import stopwords # This will help us get rid of useless words.
from keras.preprocessing import sequence
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
# Extra needed packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
IMAGE_FOLDER = os.path.join('static','img_pool')
stop_words = stopwords.words('english')
corpus = []
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model
    model = keras.models.load_model('C:/Users/LENOVO/DL_app/customer_satisfaction.h5')

#########code for review classification
@app.route('/',methods = ['GET','POST'])
def home():
    return render_template("home.html")

@app.route('/customer_satisfaction_detection',methods = ['GET','POST'])
def sent_prediction():
    if request.method=='POST':
        aaaa = []
        text=request.form['text']
        sentiment = ''
        review = text.lower()
        review = re.sub("[^a-zA-Z]", " ", review)
        review = nltk.word_tokenize(review)
        review = [word for word in review if word.lower() not in stop_words]
        lemma = WordNetLemmatizer()
        review = [lemma.lemmatize(word) for word in review]
        review  = " ".join(review)
        aaaa.append(review)
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(aaaa)
        from keras.preprocessing import sequence
        new_pred = tokenizer.texts_to_sequences(aaaa)
        pred_matrix = sequence.pad_sequences(new_pred,maxlen=150)
        probs = model.predict(pred_matrix)
        if probs < 0.2 :
            sentiment = 'Unhappy customer'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'sad_emoji.png')
        else :
            sentiment = 'Happy customer'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'happy_emoji.png')
    return render_template('home.html', review = text, sentiment = sentiment, probability = probs, image =img_filename)
##### code for customer satisfaction classification
if __name__== "__main__":
    init()
    app.run()
