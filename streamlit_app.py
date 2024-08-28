##Streamlit page app
import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import re
import numpy as np

PREDICTOR_MODEL_PATH = "SentimentAnalysis_lstm.keras"
TOKENIZER_PATH = "SentimentAnalysis_tokenizer.pickle"

predictor_model = load_model(PREDICTOR_MODEL_PATH)
tokenizer = pickle.load(open(TOKENIZER_PATH,'rb'))

print('Models loaded')

st.markdown("<h1 style='text-align: center; color: white;'>SENTIMENT CLASSIFICATION</h1>", unsafe_allow_html=True)

st.subheader("This webapp takes a phrase as input and predicts the tone of the sentence (positive or negative) and its probablity.")

st.text("")
input_text = st.text_input("Enter your phrase below for analysing the sentiment", value="")

st.text("")
st.text("")

if st.button("Predict Sentiment"):
    try:
        #Preprocess the input text
        input_text = input_text.lower()
        input_text = re.sub('[^a-zA-z0-9\s]','',input_text)
        input_text = [input_text]

        #Convert text to sequences and add padding
        sentence = tokenizer.texts_to_sequences(input_text)
        sentence = pad_sequences(sentence, maxlen=32, dtype='int32', value=0)
        
        #Make predictions on the model
        sentiment = predictor_model.predict(sentence,batch_size=1,verbose = 0)[0]
        if(np.argmax(sentiment) == 0):
            pred_sentiment = "Negative"
            pred_probablity = round(sentiment[0]*100,0)
        elif (np.argmax(sentiment) == 1):
            pred_sentiment = "Positive"
            pred_probablity = round(sentiment[1]*100,0)

        st.text("")
        st.text("")
        st.subheader(f"Predicted Sentiment: {pred_sentiment}")
        st.text(f"Probablity: {pred_probablity}%")
    except Exception as e:
        print(e)