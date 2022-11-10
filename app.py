import pickle

import numpy as np
import tensorflow
from flask import *
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from numpy.random import seed
from keras import backend as K
#setting seed value to avoid discrepancy
seed(1)
tensorflow.random.set_seed(2)


#custom function for calculating f1 score
def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


app = Flask(__name__)
model = load_model('model_lstm.h5',custom_objects={'f1':f1})




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
