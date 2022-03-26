import nltk  #Python programs to work with human language data
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer # Remove the suffix from a word and reduce it to its root.
stemmer = LancasterStemmer()

import numpy  #NumPy is a Python library used for working with arrays.It also has functions for working in domain of linear algebra, fourier transform, and matrices.
#NumPy stands for Numerical Python.
import tflearn
#TFlearn is a modular and transparent deep learning library built on top of Tensorflow. 
# It was designed to provide a higher-level API to TensorFlow in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it.
import tensorflow as tf
#The core open source library to help you develop and train ML models.
# Its flexible architecture allows for the easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.
import random
import json 
#(JavaScript Object Notation)
#JSON is a lightweight format for storing and transporting data
#JSON is often used when data is sent from a server to a web page
# JSON is "self-describing" and easy to understand
import pickle
#Python pickle module is used for serializing and de-serializing a Python object structure. 
# Any object in Python can be pickled so that it can be saved on disk.


with open("intentsss.json") as file:
    data = json.load(file)
# intents = json.loads(open('intentsss.json').read())

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []




# TO get all tag
    for intent in data["intents"]:
    # for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) # With the help of nltk.tokenize.word_tokenize() method, we are able to extract the tokens from string of characters by using tokenize.word_tokenize() method. It actually returns the syllables from a single word. A single word can contain one or two syllables
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])


        if intent["tag"] not in labels:
            labels.append(intent["tag"])
           

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()



net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=2000, batch_size=10, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=2000, batch_size=10, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Jarvis here, sir!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()    