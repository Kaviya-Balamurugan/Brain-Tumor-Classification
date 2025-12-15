# Brain Tumor Classification using MobileNetV2 and Grad-CAM

## 📌 Overview
This project implements a deep learning–based system to classify brain MRI images
into multiple tumor categories using **MobileNetV2** with transfer learning.
To enhance model interpretability, **Grad-CAM (Gradient-weighted Class Activation Mapping)**
is used to visualize the regions of the MRI images that influence the model’s predictions.

---

## 🧠 Problem Statement
Brain tumor diagnosis using MRI scans is a critical but time-consuming task that
requires expert knowledge. Manual interpretation may vary across radiologists.
This project aims to assist diagnosis by automatically classifying MRI images
using deep learning, while also providing visual explanations to improve trust
in model predictions.

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- MobileNetV2 (Transfer Learning)  
- OpenCV  
- NumPy  
- Matplotlib  

---

## 📂 Project Structure
Brain-Tumor-Classification/
├── notebooks/
│ └── brain_tumor_classification.ipynb
├── src/
│ ├── model.py
│ ├── train.py
│ └── gradcam.py
├── data/
│ └── README.md
├── results/
├── README.md
├── requirements.txt
└── .gitignore


---

## 🧪 Model Details
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Input Size: 224 × 224
- Classification Type: Multi-class
- Loss Function: Categorical Crossentropy
- Optimizer: AdamW
- Regularization: Dropout + L2

---

## 🔍 Explainability (Grad-CAM)
Grad-CAM highlights the most influential regions in MRI images that guide the
model’s predictions, improving transparency and trust.

---

## ▶️ How to Run
pip install -r requirements.txt  
python src/train.py

---

