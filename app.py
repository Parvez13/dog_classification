import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
from utils import get_classes, load_and_prep

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.h5')



def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df

class_names = get_classes()

st.set_page_config(page_title="Dog",
                   page_icon="🐶")

#### SideBar ####

st.sidebar.title("What's Dog Breed Classification  ?")
st.sidebar.write("""
Dog classification is an end-to-end **CNN Image Classification Model** which identifies the dog breed in your image. 
It can identify over 120 different dog breed
It is based upon a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Dog Dataset**.

**Accuracy :** **`80%`**

**Model :** **`EfficientNetB0`**

**Dataset :** **`Dog Breed`**
""")


#### Main Body ####

st.title("🐕‍🦺DOG BREED CLASSIFICATION🐕")
st.header("Identify what's in your dog photos!")
st.write("To know more about this app, visit [**GitHub**](https://github.com/Parvez13/dog_classification)")
file = st.file_uploader(label="Upload an image of dog.",
                        type=["jpg", "jpeg", "png"])



st.sidebar.markdown("Created by **Sohail Parvez**")


if not file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))
