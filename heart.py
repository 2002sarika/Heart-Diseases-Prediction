import numpy as np
import pickle 
import streamlit as st

load_model=pickle.load(open('D:/django/Heart Diseases/trained_model (1).sav','rb'))

#create function 
def heart_pred(input):
    input_data_as_array=np.asarray(input)
    input_data_reshaped=input_data_as_array.reshape(1,-1)
    prediction=load_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return "The person does not have heart diseases"
    else:
        return "The person have heart diseases"



def main():
         
         #give a title for web page 
        st.title("Heart Diseases Prediction") 

        #get the input from user 
        
        age=st.text_input('Age of person:')
        sex=st.text_input('Enter your gender:')
        cp=st.text_input('Chest Pain level:')
        trestbps=st.text_input("Resting_Bp value")
        chol=st.text_input('Cholestrol level:')
        fbs=st.text_input('Fasting BP value:')
        restecg=st.text_input('Resting ECG result:')
        thalach = st.text_input("Maximum Heart rate achieved:")
        exang=st.text_input("Exercised induced angina")
        oldpeak=st.text_input("ST depression level")
        slope=st.text_input("Slope of ST segment:")
        ca=st.text_input("The major vessels:")
        thal=st.text_input("Defect:")

        #code for prediction 
        diagnosis= ''

        #creating the button for prediction 
        if st.button("Heart Test Result"):
            diagnosis=heart_pred([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
        st.success(diagnosis)




if __name__ == '__main__':
    main()









