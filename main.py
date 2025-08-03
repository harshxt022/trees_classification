import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ========== Load Data and Model ==========

@st.cache_data
def load_data():
    return pd.read_pickle('tree_data.pkl')

@st.cache_resource
def load_cnn_model():
    return load_model("basic_cnn_tree_species.h5")

# ========== Main App ==========

def main():
    st.title("🌿 Tree Species Identifier (Image-based)")

    df = load_data()
    cnn_model = load_cnn_model()
    class_labels = sorted(df['common_name'].unique())

    uploaded_file = st.file_uploader("📷 Upload a tree image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        IMG_SIZE = (224, 224)
        img = image.resize(IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = cnn_model.predict(img_array)
        pred_idx = np.argmax(predictions)

        # ✅ Handle out-of-range index error
        if pred_idx >= len(class_labels):
            st.error("Predicted class index is out of range. Check model class alignment.")
            return

        pred_label = class_labels[pred_idx]
        confidence = predictions[0][pred_idx]

        st.success(f"🌳 Predicted Tree Species: **{pred_label}**")
        st.write(f"🔍 Confidence: **{confidence:.2%}**")

        st.subheader("🔝 Top 3 Predictions:")
        top_3_idx = predictions[0].argsort()[-3:][::-1]
        top_3_idx = [i for i in top_3_idx if i < len(class_labels)]
        for i in top_3_idx:
            st.write(f"{class_labels[i]} - {predictions[0][i]:.2%}")

        st.subheader(f"📌 Common Locations for '{pred_label}'")
        location_info = df[df['common_name'] == pred_label]
        if location_info.empty:
            st.warning("This species is not found in the dataset.")
        else:
            st.dataframe(
                location_info.groupby(['city', 'state'])
                .size()
                .reset_index(name='count')
                .sort_values(by='count', ascending=False)
                .head(10)
            )

if __name__ == "__main__":
    main()
