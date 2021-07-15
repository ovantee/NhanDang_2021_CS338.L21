from flask import Flask, render_template, redirect, url_for,request
import sqlite3
import os
import json
from flask import session,jsonify
import pandas as pd
import numpy as np
import joblib
import math
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.layers import Activation, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Multiply
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.layers import Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "data/"
img=''
ims=[]


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    global img,ims
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            img=image.filename
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            image=Image.open(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            image=image.resize((64,64))
            image = img_to_array(image)
            ims = [image / 255 - 0.5]
            return render_template("question.html", uploaded_image=img)
    return render_template("upload_image.html")


@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)

@app.route('/question', methods=['GET', 'POST'])
def question():
    global img,ims
    result='none'
    count_vect=pickle.load(open("model/count_vect.pkl", 'rb'))
    model_nlp=joblib.load('model/question.h5')
    show='none'
    if request.method == "POST":
        ques=[request.form['question']]
        show=str(request.form['question'])
        ques1 = count_vect.transform(ques)
        type=model_nlp.predict(ques1)[0]
        if type==0:
            model_QA=keras.models.load_model('model/Object10000_VGG_0.17_30epoch.h5')
            maxlen=33
            all_answer=pd.read_csv('model/object.csv')
            all_answer=all_answer['0'].values
            tokenizer=pickle.load(open("model/Object10000_VGG_0.17_30epoch.pkl", 'rb'))
        elif type==1:
            model_QA=keras.models.load_model('model/number_VGG_56.h5')
            maxlen=38
            all_answer=pd.read_csv('model/number.csv')
            all_answer=all_answer['0'].values
            tokenizer=pickle.load(open("model/number_VGG_56.pkl", 'rb'))
        elif type==2:
            model_QA=keras.models.load_model('model/Color10000_VGG_59.h5')
            maxlen=8
            all_answer=pd.read_csv('model/color.csv')
            all_answer=all_answer['0'].values
            tokenizer=pickle.load(open("model/Color10000_VGG_59.pkl", 'rb'))
        elif type==3:
            model_QA=keras.models.load_model('model/location_VGG_34.h5')
            maxlen=41
            all_answer=pd.read_csv('model/location.csv')
            all_answer=all_answer['0'].values
            tokenizer=pickle.load(open("model/location_VGG_34.pkl", 'rb'))
        ques=[tokenizer.texts_to_sequences(ques)[0]]
        ques = np.array(pad_sequences(ques, maxlen=maxlen,padding='pre'))
        result=all_answer[np.argmax(model_QA.predict([tf.stack(ims),ques],steps=1), axis=1)[0]]
    return render_template("question.html",rs=result,uploaded_image=img,question=show)
 
if __name__ == '__main__':
  app.run(debug = True)