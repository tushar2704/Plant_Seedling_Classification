import io
import keras
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.title("plant seedling classification")
img_size = [224, 224]

model_path = 'model.h5'
#
# model = load_model(model_path)

file = st.file_uploader('please upload an image', type=['jpg', 'png'])

class_label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
               'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',
               'Small-flowered Cranesbill',
               'Sugar beet']


def get_model(path):
    return load_model(path)


def get_image(img_path, img_size):
    img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = array / 255.0
    return array


def test():
    model = get_model(model_path)
    image = get_image(file, img_size)
    pred = model.predict(image)
    pred_index = np.argmax(pred, axis=1)
    st.write('the predicted class of image is : {}'.format(class_label[int(pred_index)]))
    print('\n')
    st.image(image, use_column_width=True)


gen_pred = st.button('Predict')
if gen_pred:
    test()
