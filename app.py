import streamlit as st 
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def text_transform(text):
    text = text.lower()
    text= nltk.word_tokenize(text)
    
    y =[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:] #clonning of list is imp bcs we cant use list again as it is mutable
    y.clear()
    
    for i in text:
        if i not in stopwords.words(text,'english') and i not in string.punctuation:
            y.append(i)  
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
   # 1. preprocess
   transformed_text = text_transform(input_sms)
   # 2. vectorize
   vector_input = tfidf.transform([transformed_text])
   # 3. predict
   result = model.predict(vector_input)[0] 
   # 4. Display
   if result == 1:
    st.header("Spam")
   else:
       st.header("Not Spam")