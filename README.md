# 🧬 Beta-Lactamase Activity Predictor (Deep Learning)
### M.Tech Thesis Project | Bioinformatics & Genomic Surveillance

## 📌 Project Overview
This repository contains the source code for a **Deep Learning-based Antimicrobial Resistance (AMR) Prediction Tool**. 
The system utilizes a **1D-Convolutional Neural Network (1D-CNN)** to analyze raw DNA sequences and predict their functional profile against four major classes of antibiotics:
1.  **Penicillinases** (Class A)
2.  **Cephalosporinases** (ESBLs)
3.  **Carbapenemases** (High-Risk Superbugs)
4.  **Monobactamases**

## 📂 Repository Structure
* `app.py` - The main Streamlit web application.
* `final_thesis_model.h5` - The pre-trained Keras/TensorFlow model (97% Accuracy).
* `requirements.txt` - Dependencies required to run the app.

## 🛠️ Technology Stack
* **Deep Learning:** TensorFlow / Keras (1D-CNN)
* **Language:** Python 3.11
* **Interface:** Streamlit
* **Data Processing:** Pandas, NumPy

## ⚙️ Local Installation
To run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ARAromal/mtech-thesis-amr.git](https://github.com/ARAromal/mtech-thesis-amr.git)
    cd mtech-thesis-amr
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📊 Model Performance
The model was trained on **3,084 DNA sequences** sourced from NCBI GenBank.
* **Validation Accuracy:** 97.21%
* **Carbapenemase Recall:** 100% (No false negatives for high-risk variants)

## 👤 Author
**A R Aromal**
M.Tech Bioinformatics
IIIT Allahabad
