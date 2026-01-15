# 🩺 Diabetic Prediction System using Gradient Boosting (GBM)

This project is an end-to-end **Machine Learning application with GUI** that predicts whether a person is **Diabetic or Not Diabetic** using the **Gradient Boosting Classifier**.  
It also includes intelligent input handling, model evaluation, and visualizations.

---

## 🚀 Project Highlights
- Gradient Boosting Classifier for diabetes prediction
- Interactive GUI built using CustomTkinter
- Intelligent medical logic (pregnancy input disabled for male patients)
- Feature Importance visualization
- Confusion Matrix visualization
- User-friendly motivational health messages
- Error handling for invalid inputs

---

## 🧠 Machine Learning Model
- **Algorithm:** Gradient Boosting Classifier  
- **Type:** Classification  
- **Evaluation Metrics:**  
  - Accuracy Score  
  - Confusion Matrix  

The model is trained on a diabetes dataset and achieves high accuracy on the test data.

---

## 🖥️ GUI Features
- Gender selection using dropdown
- Automatic handling of pregnancy input:
  - Male → Pregnancies set to 0 (not asked)
  - Female → Pregnancy input enabled
- Clean and responsive UI
- Displays prediction result with health guidance
- Shows trained model accuracy

---

## 📊 Visualizations
- **Feature Importance Plot**  
  Shows which medical factors contribute most to prediction.
- **Confusion Matrix**  
  Displays actual vs predicted classification results.

> Note: Visualizations open in a separate Matplotlib window.

---

## 📁 Project Structure
  DIABETIC_PREDICTION_GBM/
  │── main.py / app.py
  │── diabetes_prediction_dataset_file.csv
  │── README.md
  │── requirements.txt

## 📦 Requirements
  Python 
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  customtkinter

## 🧪 Sample Prediction Output

  🟢 Not Diabetic
  Keep up the healthy lifestyle!
  
  🔴 Diabetic
  Early detection helps. Please consult a doctor for guidance.
  
##  📌 Disclaimer
  This project is for educational purposes only and should not be used as a substitute for professional medical advice.

###########################################
## 👨‍💻 Author
  Yash Garg
