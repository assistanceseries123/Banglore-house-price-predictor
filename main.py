import streamlit as st
import pickle
import sklearn
import numpy as np
import pandas as pd
st.set_page_config(layout='centered',page_title='Banglore house price predictor')


pipe=pickle.load(open('Ridgemodel.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title("Banglore house price predictor")

#location
location=st.selectbox('Location',df['location'].unique())

#Total square ft
squarefoot=st.selectbox("Total square feet",df['total_sqft'].unique())

#bathroom

bathroom=st.selectbox('Bathroom',df['bath'].unique())

#bed room hall kitchen
bedroom=st.selectbox('Bedroom Hall Kitchen',df['bhk'].unique())

if st.button('Predict price'):

    
    query_df = pd.DataFrame([[location,squarefoot,bathroom,bedroom]],columns=['location', 'total_sqft', 'bath', 'bhk'])
    st.subheader("Therefore The Price Of The House Is In Lakh")
    st.title("The predicted price of this configuration is RS:" + str(int(pipe.predict(query_df)[0])))