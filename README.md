# Chatbot Project

## Introduction

Chatbots have become an essential tool for businesses and customers alike. Many users prefer interacting with chatbots rather than calling customer service centers, as chatbots provide instant and convenient support. Facebook data shows that over 2 billion messages are exchanged monthly between people and companies via bots. According to HubSpot, 71% of people prefer getting customer support through messaging apps, highlighting the growing importance and future potential of chatbots in organizations.

This project demonstrates how to build a chatbot from scratch capable of understanding user inputs and providing relevant responses using deep learning and natural language processing (NLP) techniques.

---

## How Chatbots Work

Chatbots are intelligent software that interact with users similarly to humans. They are based on **Natural Language Processing (NLP)**, which involves:

* **Natural Language Understanding (NLU):** The ability to comprehend human language.
* **Natural Language Generation (NLG):** The ability to generate human-like responses.

For example, if a user says, “Hey, what’s on the news today?”, the chatbot extracts:

* **Intent:** The user's goal (e.g., `get_news`).
* **Entity:** Specific details related to the intent (e.g., `today`).

Our chatbot uses a machine learning model to classify these intents and entities from user messages to generate suitable responses.

---

## Prerequisites

Make sure you have Python installed. Install the required libraries using:

```bash
pip install tensorflow keras pickle nltk
```

---

## Project Structure

* **train\_chatbot.py** — Trains the deep learning model to classify user intents.
* **gui\_chatbot.py** — Implements the graphical user interface (GUI) to interact with the chatbot.
* **intents.json** — Contains tagged patterns and corresponding responses for training.
* **chatbot\_model.h5** — Saved trained model weights and architecture.
* **classes.pkl** — Pickle file storing intent tags.
* **words.pkl** — Pickle file storing the vocabulary of unique words.

---

## Step-by-Step Guide to Build the Chatbot

### Step 1: Import Libraries and Load Data

We import essential libraries like Keras for deep learning, NLTK for NLP, and pickle for saving data. Then we load the `intents.json` file which contains the data to train the chatbot.

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
```

---

### Step 2: Preprocessing the Data

We tokenize sentences into words, lemmatize (reduce words to their base form), and build a vocabulary (`words`) and a list of classes (intents).

```python
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
```

---

### Step 3: Creating Training Data

We create training data where each pattern is converted to a bag-of-words representation (a binary array indicating presence of words), and the output is a one-hot encoding of the class.

```python
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
```

---

### Step 4: Training the Model

We define a neural network with 3 dense layers and train it using the Stochastic Gradient Descent optimizer for 200 epochs.

```python
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
```

---

### Step 5: Interacting with the Chatbot (GUI)

Using Tkinter, we build a desktop interface that captures user input, processes it, predicts the intent, and returns a random response from the corresponding intent.

Key functions:

* **clean\_up\_sentence:** Tokenizes and lemmatizes the user input.
* **bag\_of\_words:** Converts the input sentence into a bag-of-words array.
* **predict\_class:** Predicts the intent using the trained model.
* **getResponse:** Selects a random response based on predicted intent.

Example snippet for the GUI:

```python
import tkinter as tk
from keras.models import load_model
import nltk
import pickle
import json
import random

# Load required data and model
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = nltk.WordNetLemmatizer()

# Define functions ...

# Setup Tkinter GUI components and mainloop ...
```

---

## Conclusion

This project demonstrates how to build a fully functioning chatbot with deep learning and NLP techniques. You can extend this project by:

* Adding more intents and patterns to `intents.json`.
* Improving preprocessing with advanced NLP techniques.
* Deploying the chatbot on web or mobile platforms.

Feel free to explore, modify, and expand this chatbot to suit your needs!

---

If you want me to help generate the full GUI code or any other part formatted nicely, just ask!
