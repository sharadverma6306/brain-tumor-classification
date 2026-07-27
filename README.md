# 🧠 Brain Tumor Classification using Deep Learning

An end-to-end Deep Learning web application that classifies Brain MRI images into one of four categories using a Convolutional Neural Network (CNN). The application is built with **TensorFlow/Keras** and deployed using **Streamlit** for an interactive user experience.

> **⚠️ Disclaimer:** This project is developed for educational purposes only and should **not** be used as a medical diagnostic tool.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://brain-tumor-classification-w3pmwpprlw8dc5imdtsxqy.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/sharadverma6306/brain-tumor-classification

---

## 📌 Features

- Upload Brain MRI images (.jpg/.png)
- Classifies MRI images into four categories
- Displays prediction confidence score
- Shows prediction probability distribution
- Provides basic information about the predicted tumor
- Interactive Streamlit web application

---

## 🧠 Tumor Classes

The model classifies Brain MRI images into the following categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pillow

---

## 📁 Project Structure

```text
brain-tumor-classification/
│
├── assets/
├── model/
│   └── brain_tumor_model.keras
├── app.py
├── predict.py
├── train.py
├── tumor_info.py
├── requirements.txt
├── runtime.txt
└── .gitignore
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/sharadverma6306/brain-tumor-classification.git
cd brain-tumor-classification
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📊 Model

- Convolutional Neural Network (CNN)
- Built using TensorFlow/Keras
- Trained on Brain MRI images
- Saved in `.keras` format

---

## 🎯 How It Works

1. Upload a Brain MRI image.
2. The image is preprocessed.
3. The trained CNN model predicts the tumor type.
4. The application displays:
   - Predicted class
   - Confidence score
   - Prediction probabilities
   - Basic information about the predicted tumor

---

## 📦 Requirements

Main dependencies:

- tensorflow-cpu
- streamlit
- numpy
- opencv-python
- pillow
- matplotlib

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Sharad Verma**

- GitHub: https://github.com/sharadverma6306
- LinkedIn: https://www.linkedin.com/in/sharadverma6306/

---

## 🙏 Acknowledgements

Special thanks to **GRASTech** and all the mentors for their guidance and support throughout this project.

---

## ⭐ Support

If you found this project helpful, consider giving it a **Star ⭐** on GitHub.