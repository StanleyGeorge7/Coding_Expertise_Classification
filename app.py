import pickle

import numpy as np
import tensorflow
from flask import *
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from numpy.random import seed

#setting seed value to avoid discrepancy
seed(1)
tensorflow.random.set_seed(2)

app = Flask(__name__)
model = load_model('model_lstm.h5')

@app.route('/')
def homescreen():
    return render_template('frontend.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    message = request.form['message']
    corpus = [message]
    corpus=list(corpus)
    sent_length=20
    onehot_repr=[one_hot(words,50)for words in corpus] 
    embedded_docs_test= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    prediction=model.predict(embedded_docs_test)
    predicted_labels=[np.argmax(i) for i in prediction]
    print(predicted_labels)
    predicted_labels="".join(str(x) for x in predicted_labels)
    predicted_labels=int(predicted_labels)
    if predicted_labels==0:
        print(predicted_labels)
        return render_template('frontend.html',pred='''He/She is newbie to Programming''', x="")
    elif predicted_labels==1:
        print(predicted_labels)
        return render_template('frontend.html',pred='''He/She has Average Programming Skills''', x="")        
    else:
        print(predicted_labels)
        return render_template('frontend.html', pred='He/she has a good Programming knowledge ''', x="")


if __name__ == "__main__":
    app.run(debug=True)
