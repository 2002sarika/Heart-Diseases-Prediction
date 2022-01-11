import numpy as np
import pickle 

load_model=pickle.load(open('D:/django/Heart Diseases/trained_model (1).sav','rb'))

input=(63,1,3,145,233,1,0,150,0,2.3,0,0,1)
input_data_as_array=np.asarray(input)
input_data_reshaped=input_data_as_array.reshape(1,-1)
prediction=load_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print("The person is does not have heart diseases")
else:
  print("The person have heart diseases")