# AI/ML Engineering – Advanced Internship Tasks

**Name:** Alishba Noor  

**Intern ID:** DHC-1539  

**Internship Status:** Ongoing  

**Assigned Task:** Task 2 (Categorized into 5 subtasks)  

**Subtasks Completed:** 1, 2, 3  



## **Overview**
This repository contains completed tasks from the Advanced Internship Program at **DevelopersHub Corporation**. These tasks are designed to provide hands-on experience with modern machine learning and artificial intelligence techniques, including transformer models, machine learning pipelines, multimodal learning, and deep learning.

**Completed Tasks:**  
- Task 1: News Topic Classifier Using BERT  
- Task 2: End-to-End ML Pipeline with Scikit-learn  
- Task 3: Multimodal Housing Price Prediction  

**Technologies Used:**  
Hugging Face Transformers, Scikit-learn, TensorFlow/Keras, Gradio, joblib, CNNs, Pandas

---

## **Task 1: News Topic Classifier Using BERT**

**Objective:**  
Fine-tune a BERT model to classify news headlines into topic categories using the AG News dataset.

**Methodology:**  
- Loaded and tokenized the AG News dataset using Hugging Face Datasets and Tokenizers  
- Used `bert-base-uncased` and fine-tuned it using the `Trainer` API  
- Evaluated the model using **accuracy** and **F1-score**  
- Deployed the model using **Gradio** for interactive predictions

**Key Results:**  
- Achieved high classification accuracy and F1-score on validation data  
- Built a deployable and reusable model for topic classification

---

## **Task 2: End-to-End ML Pipeline with Scikit-learn**

**Objective:**  
Build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn dataset.

**Methodology:**  
- Preprocessed data by handling missing values, scaling numerical features, and encoding categorical features  
- Used `ColumnTransformer` and `Pipeline` for clean preprocessing and modeling  
- Trained **Logistic Regression** and **Random Forest** classifiers  
- Performed hyperparameter tuning with **GridSearchCV**  
- Exported the best model pipeline using **joblib**

**Key Results:**  
- Built a fully encapsulated and production-ready ML pipeline  
- Achieved high accuracy on training data  
- Model is ready for deployment and inference

---

## **Task 3: Multimodal ML – Housing Price Prediction**

**Objective:**  
Predict house prices using both structured tabular data and image data.

**Methodology:**  
- Loaded tabular features and normalized them using `StandardScaler`  
- Extracted image features using a frozen **MobileNetV2** CNN backbone  
- Combined tabular and image features using concatenation layers  
- Trained a regression model using TensorFlow/Keras  
- Evaluated model performance using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**

**Key Results:**  
- Successfully built a multimodal architecture that combines CNN and tabular features  
- Achieved competitive MAE and RMSE scores on test data  
- Demonstrated practical use of deep learning in multimodal scenarios

