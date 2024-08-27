import re
import pandas as pd
import streamlit as st
import nltk
from sklearn.model_selection import train_test_split

import pandas as pd 
import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
#from imblearn.over_sampling import SMOTE
import tensorflow


class CustomLSTM(keras.layers.LSTM):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)  # Ignore the time_major argument if it's passed
        super().__init__(**kwargs)

model =load_model('D:\programeing labs\MLintrn\Intern\dataset\sentiment_lstm.h5', custom_objects={'LSTM': CustomLSTM})
def cleaning(text):
  text=re.sub(r"@[a-zA-Z0-9]+",'',text)
  text=re.sub(r"@[A-Za-zA-Z0-9]+",'',text)
  text=re.sub(r"@[a-zA-Z]+",'',text)
  text=re.sub(r"@[-)]+",'',text)
  text=re.sub(r"#",'',text)
  text=re.sub(r"RT[\s]+",'',text)
  text=re.sub(r"http?\/\/\s+",'',text)
  return text

stop=set(stopwords.words("english"))
lemitizer=WordNetLemmatizer()
nltk.download("punkt_tab")
def limatize(text):
    tokens=word_tokenize(text)
    words=[lemitizer.lemmatize(word) for word in tokens if word.lower() not in stop]
    return ' '.join(words)
st.title("Sentimental Analysis")
name=st.text_input("enter the sentiments")

user_input=pd.Series(name)
data=user_input.apply(cleaning)

data=data.apply(limatize)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_vocab = 10000

# Initialize tokenizer
tokenizer = Tokenizer(num_words=max_vocab)

# Ensure tokenizer is not None (no need for the if-check here)
if tokenizer is not None:
    tokenizer.fit_on_texts(name)
    s=tokenizer.word_index
    v=len(s)
      # This fits the tokenizer, it does not return anything
else:
    print("Tokenizer is None")

# Convert text to sequences
train_seq = tokenizer.texts_to_sequences(user_input)


# Check if the tokenizer is none
if tokenizer is None:
    print("Tokenizer is None")
else:
    print("Tokenizer is working correctly")
pad_train=pad_sequences(train_seq,maxlen=163)
t=pad_train.shape[1]
result=[[0,0],"dd"]
result=model.predict(pad_train)
import numpy as np

# Example output from the model
prediction = np.array(result[0])

# Softening transformation
def soften_probabilities(values, alpha=0.1):
    return 1 / (1 + np.exp(-alpha * (values - 0.5)))

# Apply the transformation
softened_prediction = soften_probabilities(prediction)



# Define thresholds
# threshold_1 = 0.5  # Threshold for the first probability
# threshold_2 = 0.5  # Threshold for the second probability

# Define function to classify based on thresholds
def classify_output(pred):
    if ((pred[0]-pred[1])*(-1)<=0.1):
      st.write('neutral')
    elif pred[1]>pred[0]:
        st.write('Class 1 (posistive)')
    elif pred[1]>pred[0]:
        st.write('Class 2 (negetive)')
    else:
        st.write('Class 3 (irrelevent)')


# Apply the function to classify the output
sentiment = classify_output(softened_prediction)

# Output the sentiment
print("Predicted Sentiment:", sentiment)
st.write(sentiment+"hhj")


