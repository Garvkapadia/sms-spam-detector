
import streamlit as st
import pickle



tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

import string
import nltk

nltk.download('stopwords', quiet=True, download_dir='/opt/render/nltk_data')
nltk.download('punkt_tab', quiet=True, download_dir='/opt/render/nltk_data')
nltk.download('punkt', quiet=True, download_dir='/opt/render/nltk_data')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def transform_text(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))

    return " ".join(y)
st.title('SMS Spam Detection System')
input_sms=st.text_input("Enter the message:")
if st.button('Predict'):
 transformed_sms=transform_text(input_sms)
 input_vector=tfidf.transform([transformed_sms])
 result=model.predict(input_vector)[0]
 if result==0:
    st.header('NOT SPAM')
 else:
    st.header('SPAM')