import os

from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import streamlit as st
from groq import Groq


os.environ["GROQ_API_KEY"] = "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"
def fetchrec(classes):
    prompt = f"Recommendation for patient with {classes} brain tumour. Remove the ai part and give only relevant details."
    try:
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )


        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        res = chat_completion.choices[0].message.content.strip()
        return res

    except Exception as e:
        return str(e)


model = load_model("braintumourclassifier3rd.h5")
classes = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

st.title("Brain Tumor Classification and Recommendation App")

uploaded_file = st.file_uploader("Choose an image from your device", type="jpg")

if uploaded_file is not None:
    # Preprocess
    uploaded_image = Image.open(uploaded_file)

    if uploaded_image.mode != "RGB":
        uploaded_image = ImageOps.grayscale(uploaded_image).convert("RGB")

    uploaded_image = uploaded_image.resize((128, 128))

    img_array = img_to_array(uploaded_image)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize
    img_array /= 255.0

    # Prediction
    prediction = model.predict(img_array)
    predict_index = np.argmax(prediction)
    predicted_class = classes[predict_index]

    recommendation = fetchrec(predicted_class)

    # Result
    st.image(uploaded_image, use_column_width=True)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Recommendation : {recommendation}")


st.write("Built By Tushar Sharma")
st.markdown("[GitHub Profile](https://github.com/Tusshaar28)")

