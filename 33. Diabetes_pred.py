# DIABETES PREDICTION (classification)
# using GRADIENT BOOSTING

# Diabete prediction function: is an numeric value that represent likelihood of diabetes based on family history
# rannge = 0.1 to 2.5
# 0.1-0.5 = Low family history risk
# 0.6-1.0 = Moderate Risk
# 1.1-2.5 = High Risk

#----------IMPORT MODULES----------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from  sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
#----------LOAD DATSET-------------
df=pd.read_csv("E:\Desktop\ML\GRADIENT_BOOSTING\Diabetes_GBM\diabetes_prediction_dataset_file.csv")
print(df)

#-----ENCODING----------
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender']) # Male=1 , Female=0
print(df)

#-----FEATURE--------
x=df.drop('Outcome',axis=1) # Indepenent
y=df['Outcome'] #  Dependent

#-----SPLIT_DATA----------
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#-----Load Model-----------
model=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,random_state=42)

#-----TRAIN_MODEL------------
model.fit(x_train,y_train)

#----TEST_PREDICTION----------
y_test_pred=model.predict(x_test)

#----EVALUATION (classification)---------
acc=accuracy_score(y_test,y_test_pred)
cfm=confusion_matrix(y_test,y_test_pred)

print("TEST_Model_Evaluation:")
print(f'Accuracy:', acc) # 1.0

print("Confusion_Matrix:") 
print(cfm) 
# [[60  0]
#  [ 0 40]]


# #-----MANUAL PREDICITION (without GUI) -------------
# # take USER input
# print("Enter Patient details to Predict Diabetes\n {0=Not Diabetic,1=Diabetic}")

# gender=input("Enter Gender {Male/Female}: ").strip().capitalize()
# if gender=='Female':
#     Pregnancies=int(input("Number of Pregnancies: "))
#     gender_encode=0
# else:
#     gender_encode=1
#     Pregnancies=0

# Glucose = int(input("Glucose Level: "))
# BloodPressure = int(input("Blood Pressure: "))
# SkinThickness = int(input("Skin Thickness: "))
# Insulin = int(input("Insulin Level: "))
# BMI = float(input("BMI: "))
# DPF = float(input("Diabetes Pedigree Function: "))
# Age = int(input("Age: "))


# # Create input DataFrame
# input_data = pd.DataFrame([{
#     "Pregnancies": Pregnancies,
#     "Glucose": Glucose,
#     "BloodPressure": BloodPressure,
#     "SkinThickness": SkinThickness,
#     "Insulin": Insulin,
#     "BMI": BMI,
#     "DiabetesPedigreeFunction": DPF,
#     "Age": Age,
#     "Gender": gender_encode
# }])

# result=model.predict(input_data)[0]
# print(f"\n Prediction: {'Diabetic' if result == 1 else 'Not Diabetic'}")

################################################################3
#--------GUI-------------------------------------
import customtkinter as ctk 

root=ctk.CTk()
root.title("Diabetes Prediction System")
root.geometry("450x600")

title = ctk.CTkLabel(root, text="Diabetes Prediction", font=("Arial", 22, "bold"))
title.pack(pady=20)

# ------------------ FUNCTIONS ------------------
def toggle_pregnancy_field(choice):
    if choice == "Female":
        pregnancy_entry.pack(pady=5)
    else:
        pregnancy_entry.pack_forget()

def predict_result():
    try:
        gender = gender_var.get()

        if gender == "Male":
            pregnancies = 0
            gender_encode = 1
        else:
            pregnancies = int(pregnancy_entry.get())
            gender_encode = 0
        
        gl=int(fields['Glucose'].get())
        bp=int(fields['BloodPressure'].get())
        st=int(fields['SkinThickness'].get())
        insl=int(fields['Insulin'].get())
        bmi=float(fields['BMI'].get())
        dpf=float(fields['DiabetesPedigreeFunction'].get())
        age=int(fields['Age'].get())
        gender=gender_encode

        feature=[[pregnancies,gl,bp,st,insl,bmi,dpf,age,gender]]

        prediction = model.predict(feature)[0]


        if prediction == 1:
            result_label.configure(text="🔴 Result: Diabetic\n🩺 Please consult a doctor for guidance.", text_color="red",font=("Arial", 20))
            acc_label.configure(text=f"Model Accuracy : {acc*100:.2f}%",font=("Arial", 20),text_color='blue')
        else:
            result_label.configure(text="🟢 Not Diabetic\n🌟 Keep up the healthy lifestyle", text_color="green",font=("Arial", 20))
            acc_label.configure(text=f"Model Accuracy : {acc*100:.2f}%",font=("Arial", 20),text_color='blue')

    except ValueError:
        # messagebox.showerror("Invalid Input", "Please enter valid numerical values.")
        result_label.configure(text="Please enter valid numerical values.",fg='red',font=("Arial", 20),text_color='black')

# Plot function
def plot_fi():
        # feature_importance 
        fei=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
        plt.figure(figsize=(8,5))
        ax=sns.barplot(y=fei,x=fei.index,palette='viridis')
        for container in ax.containers:
            ax.bar_label(container)
        plt.xlabel("Feature_Importance")
        plt.ylabel("Features")
        plt.title("FEATURE IMPORTANCE PLOT\n\n'BMI' & 'GLUCOSE'\ncontributes the most to predictions.")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
# plot confusion matrix
def plot_cm():
    mydict={'fontsize':33,'fontstyle':'italic','color':'k','weight':0.5,'verticalalignment':'center'}
    sns.heatmap(cfm, annot=True, cmap='coolwarm',annot_kws=mydict,linewidths=2,linecolor='y', yticklabels=['Not Diabetic', 'Diabetic'],xticklabels=['Not Diabetic', 'Diabetic'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()



#------------FRAME-1--------------------------------
toggle_frame=ctk.CTkFrame(root)
toggle_frame.pack()

gender_var = ctk.StringVar(value="Male")
gender_menu = ctk.CTkOptionMenu(toggle_frame,values=["Male", "Female"],variable=gender_var,width=180,command=toggle_pregnancy_field)
gender_menu.pack(pady=5)

# Pregnancy (hidden initially)
pregnancy_entry = ctk.CTkEntry(toggle_frame,placeholder_text="Number of Pregnancies",width=180,font=("Arial", 15))

#------Frame-2----------------------------------
entry_frame=ctk.CTkFrame(root)
entry_frame.pack()

# Entry labels and fields
fields={
        "Glucose": None,
        "BloodPressure": None,
        "SkinThickness":None,
        "Insulin":None,
        "BMI": None,
        "DiabetesPedigreeFunction": None,
        "Age":None }

for i,label in enumerate(fields):
    entry=ctk.CTkEntry(entry_frame,placeholder_text=label,width=180,font=("Arial", 15))
    entry.grid(row=i,column=0,pady=5)
    fields[label]=entry

#---------------------------------------------------------------
# Result Label
result_label = ctk.CTkLabel(root, text="")
result_label.pack(pady=5)

# Predict Button
predict_btn = ctk.CTkButton(root, text="Predict Diabetes",font=("Arial", 20,'bold'),command=predict_result)
predict_btn.pack(pady=5)

# accuracy label
acc_label = ctk.CTkLabel(root, text="")
acc_label.pack(pady=5)

ctk.CTkButton(root, text="Feature_Importance", command=plot_fi,
          font=("Arial",20,'bold')).pack(pady=5)
ctk.CTkButton(root, text="Confusion_Matrix", command=plot_cm,
          font=("Arial",20,'bold')).pack(pady=5)
root.mainloop()





