# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 10:44:56 2022

@author: HP
"""

import pickle
import pandas as pd
import numpy as np
from statistics import mode
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder

# Loading the Saved Model
diabetes_model   = pickle.load(open('E:/Projects/Diseases Prediction/Saved Model/diabetes_model.sav', 'rb'))
heart_model      = pickle.load(open('E:/Projects/Diseases Prediction/Saved Model/heart_disesases_model.sav', 'rb'))
parkinsons_model = pickle.load(open('E:/Projects/Diseases Prediction/Saved Model/parkinson_model.sav', 'rb'))
final_rf_model   = pickle.load(open('E:/Projects/Diseases Prediction/Saved Model/DPS_RF_model.sav', 'rb'))
final_nb_model   = pickle.load(open('E:/Projects/Diseases Prediction/Saved Model/DPS_NB_model.sav', 'rb'))
final_svm_model  = pickle.load(open('E:/Projects/Diseases Prediction/Saved Model/DPS_SVM_model.sav', 'rb'))

data = pd.read_csv('E:/Projects/Diseases Prediction/Dataset Files/Training_dps.csv')
data.drop(columns='Unnamed: 133', inplace=True)

encoder = LabelEncoder()
data['prognosis'] = encoder.fit_transform(data['prognosis'])
X = data.drop(columns='prognosis')


# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptoms = X.columns.values
sym = symptoms
 
# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
 
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}
 
# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
     
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
     
    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": nb_prediction,
        "final_prediction":final_prediction
    }
    print(final_prediction)
    return rf_prediction

count = 0
temp =''
sym.sort()
sym = sym.tolist()
sym.insert(0,'Select')

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diseases Prediction Symptoms',
                           'Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    

# Diseases Prediction From Symptoms
if (selected == 'Diseases Prediction Symptoms'):
    
    # Page title
    st.title('Diseases Prediction From Symptoms')
    
    sympt =[]
    col1, col2 = st.columns(2)
    
    with col1:
       Symptom1 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
       
    with col2:
       Symptom2 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
       
    with col1:
       Symptom3 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
       
    with col2:
       Symptom4 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
       
    with col1:
       Symptom5 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
       
    with col2:
       Symptom6 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
       
    with col1:
       Symptom7 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
       
    with col2:
       Symptom8 = st.selectbox('Select Symptoms From List', sym, key=count)
       count += 1
    
    final_result =''
    if st.button('Predict Diseases'):
        sympt = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, Symptom6]
        
        sympt = [i for i in sympt if i != 'Select']
        for str in sympt:
            temp = temp + str.title().replace("_"," ") + ','
        
        temp = temp[:-1]
        
            
        if len(sympt) < 5:
            final_result = 'Please Give More Input'
        else:
            final_result = predictDisease(temp)
                
    st.success(final_result)


# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

