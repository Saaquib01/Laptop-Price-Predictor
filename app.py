import streamlit as st
import pickle
import numpy as np

# model import
pipe =  pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("LAPTOP PREDICTOR")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type
laptop_type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(In GB)',[2,4,6,8,12,16,24,32,64])

# Weight
Weight = st.number_input('Weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['Yes','No'])

# IPS 
ips = st.selectbox('IPS',['Yes','No'])

# Screensize 
Screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Resolution',['1920x1080','1366x768','1600x900','3840x2160','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
Cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD(In GB)',[0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox('SSD(In GB)',[0,8,128,256,512,1024])

# gpu
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

#os
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen =='Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+(Y_res**2))**0.5/Screen_size
    query = np.array([company,laptop_type,ram,Weight,touchscreen,ips,ppi,Cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("Predicted Price: $" + str(np.exp(pipe.predict(query))[0]))
