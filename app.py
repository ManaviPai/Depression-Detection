import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
#model = load_model('model.h5')
import json
import random
import pandas as pd
import re
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer
token = Tokenizer()
from keras.preprocessing.sequence import pad_sequences
import time
import numpy as np


intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

global model
model = load_model("model.h5")
model._make_predict_function()
print("Model Loaded")
graph = tf.get_default_graph()

global emo_model 
emo_model = load_model("cnn_model.h5")
emo_model._make_predict_function()
print("Model Loaded")
graph1 = tf.get_default_graph()

emotions= []
data_train = pd.read_csv('/data/data_train.csv', encoding='utf-8')
data_test = pd.read_csv('/data/data_test.csv', encoding='utf-8')
data = data_train.append(data_test, ignore_index=True)

def clean_text_emotion(data):
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    
    # tekenization using nltk
    data = word_tokenize(data)
    
    return data

texts = [' '.join(clean_text_emotion(text)) for text in data.Text]
token.fit_on_texts(texts)

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    with graph.as_default():
    	res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def calc():
    neg = 0 
    pos = 0
    for i in emotions:
            if i == 'sadness':
                neg = neg+1
            elif i== 'fear':
                neg = n+1
            elif i == 'anger':
                neg = neg+1
            elif i == 'neutral':
                pos = pos+1
            else:
                pos = pos+1
    np = neg*100/(pos+neg)
    if np<20:
        return "You are neither stressed or depressed. That's awesome! You should remain happy like this!"
    elif np>=20 and np<40:
        return "You are slighlty stressed. Take it a little easy and go for walks. Clear your head! It'll help relieve the stress"
    elif np>=40 and np<60:
        return "You are extremely stressed. Maybe it is just a phase you are going through but breathe and take it a little easy. Go for a walk and do things that might calm you"
    elif np>=60 and np<75:
        return "Oh no! You are in slight depression! Let's try to overcome this? Please talk to someone about it. If you need to speak to a therapist, let me know. It's good to speak to a therapist. It relieves all those thoughts going on in your head"
    else:
        return "You are in depression. Let's try to overcome this? Please talk to someone about it. If you need to speak to a therapist, let me know. It's good to speak to a therapist. It relieves all those thoughts going on in your head"


def cnn_model_predict(msg):
    class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
    msg1 = [msg]
    seq = token.texts_to_sequences(msg1)
    padd = pad_sequences(seq, maxlen=500)
    with graph1.as_default():
        pred = emo_model.predict(padd) 
    emotions.append(class_names[np.argmax(pred)])
    print(emotions)
    return(class_names[np.argmax(pred)])


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    cnn_model_predict(userText)
    if chatbot_response(userText) == 'Great!':
        return calc()
    else:
        return chatbot_response(userText)


if __name__ == "__main__":
    app.run()
