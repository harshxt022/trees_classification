# trees_classification
# 🌿 Tree Species Identifier

A deep learning-powered web app that identifies tree species from images and suggests their most common geographic locations. Built with **TensorFlow**, **Streamlit**, and **Pandas**.

---

## 🚀 Features

- 📷 **Image-Based Tree Identification**  
  Upload a photo and get instant predictions of the tree species using a trained CNN model.

- 📌 **Location Insights**  
  Find the top cities and states where a predicted tree species is commonly found (based on real-world dataset).

- 🧠 **Trained on Real Tree Data**  
  The model is trained on a curated dataset of 30+ Indian tree species with location tags.

---

## 🔧 Tech Stack

| Tool            | Purpose                         |
|-----------------|----------------------------------|
| `TensorFlow`    | Deep learning model (CNN)        |
| `Streamlit`     | Web app interface                |
| `Pandas`        | Data handling and location stats |
| `Keras`         | Model training and prediction    |
| `.h5 Model`     | Saved trained CNN model          |

---

## 📁 Files

| File                         | Description                         |
|------------------------------|-------------------------------------|
| `app.py` / `main.py`         | Streamlit frontend code             |
| `basic_cnn_tree_species.h5`  | Trained CNN model (image classifier)|
| `tree_data.pkl`              | Pickled dataset with tree info      |
| `scaler.joblib` + `nn_model.joblib` | For extra features (optional) |

---

## 🧪 Demo

Upload a tree image and see:

- ✅ Predicted species  
- 🎯 Confidence level  
- 📍 Locations where this species is common

![demo screenshot](demo_screenshot.png) <!-- Replace with actual screenshot -->

---

## ⚠️ Note

- GitHub doesn’t support files >100MB. Use **Git LFS** or external links (e.g., Google Drive) for the `.h5` model.
- Make sure `tree_data.pkl` contains real species and location data.

---

## ✨ Future Enhancements

- 📈 Accuracy improvement using **EfficientNet** or **ResNet**
- 🌍 Map-based location visualizations
- 📚 Include leaf shape, size, and other features as inputs

---

## 👨‍💻 Author

Built with 💚 by [Harshit Hassija](https://github.com/harshxt022)

---

## 🪴 License

MIT License – feel free to fork, learn, and grow 🍃
