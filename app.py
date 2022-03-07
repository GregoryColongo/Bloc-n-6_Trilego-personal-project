# Import des librairies:

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model


###  En-tête de notre application:  ###
img = Image.open("lego3.png")

st.set_page_config(page_title= "Projet Jedha", page_icon= img)

image = Image.open('lego.png')

st.image(image, caption='Projet TriLEGO')

st.header("Class your Lego")
st.subheader('To which setbox this brick belongs to ?')

hide_menu_style = """
           <style>
            #MainMenu {visibility: shown;}
            footer {visibility: hidden;}
            </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True )

###  Première fonction:  ###
def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_image(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [224, 224])

def giveset(dataframe, pred):
    y = dataframe["Set"][dataframe["Ref"] == int(pred)].values[0]
    
    if y == "41367" :
        st.text("The brick belongs to the setbox : ")
        st.image("SET2.png")
    
    else:
        st.text("The brick belongs to the setbox : ")
        st.image("SET1.png")  
    
    return y

def predict_class(model, img):
                                            
    test_image = np.expand_dims(img, axis = 0)                          
    class_names = ["300223","403221","4220631","6151578","6172421","6210741"]  
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    
    return image_class

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    classifier_model = tf.keras.models.load_model(model_name)
    return classifier_model
classifier_model = load_model("Xception224_model.h5")   

def main():

    file_uploaded = st.file_uploader(" LEGO brick to identify : ", type = ["jpg", "jpeg", "png"])
    
    if file_uploaded is not None :                      
     
        img = decode_img(file_uploaded.read())
        img = img/255

        predicted_class = predict_class(classifier_model,img)
        match_set = giveset(pd.read_csv("./bdd/Dataset_ref.csv", delimiter = ";"),predicted_class)

        st.text(" percentage :" + str(match_set))
        print(match_set)
        
      
if __name__ == "__main__":
    main()




