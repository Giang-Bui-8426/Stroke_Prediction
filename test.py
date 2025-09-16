import joblib
import pandas as pd
import numpy as np
import tkinter as tk
import joblib

best_model = joblib.load("stroke_model.pkl")
processor = joblib.load("processor.pkl")
encoders = joblib.load("encoders.pkl")
data = pd.read_csv("stroke_data.csv")
col = data.drop(["stroke","ever_married"],axis=1).columns
All_col = joblib.load("all_cols.pkl")
threshold  = joblib.load("threshold.pkl")
def predict():
    new_data = pd.DataFrame([{
        "gender": gender_var.get(),
        "age": int(age.get()) if age.get() else None,
        "hypertension": 1 if hypertension_var.get() == "Yes" else 0,
        "work_related_stress": work_var.get(),
        "urban_residence": urban_var.get(),
        "smokes": smokes_var.get() ,
        "avg_glucose_level":float(glucose.get()) if glucose.get() else None,
        "bmi" : float(bmi.get()) if bmi.get() else None,
        "heart_disease" : 1 if Heart_var.get() == "Yes" else 0,
    }])
    data_predict = clean_missData(new_data,col)
    data_predict = processor.transform(data_predict)
    data_predict = pd.DataFrame(data_predict, columns=All_col)
    proba = best_model.predict_proba(data_predict)[0]
    stroke.delete(0, tk.END)
    if proba[1] > threshold:
        stroke.insert(0,1)
    else:
        stroke.insert(0, 0)
    print("P(No) = %.3f, P(Yes) = %.3f" % (proba[0], proba[1]))
    # Nếu có dùng LabelEncoder cho y, giải mã lại
    # pred1 = "Yes" if encoders["stroke"].inverse_transform(y_pred)[0] == 1 else "No"
    # stroke.delete(0, tk.END)
    # stroke.insert(0,pred1)

def clean_missData(data,col_data):
    for i in  col_data:
        if i not in data:
            data[i] = np.nan
    return data[col_data]


root = tk.Tk()
root.title("Stroke Prediction")
root.geometry("400x500")
tk.Label(root, text="Age:").grid(row=1, column=0, padx=10, pady=5, sticky="e", )
age = tk.Entry(root)
age.grid(row=1, column=1, padx=10, pady=5)
tk.Label(root, text="Avg glucose level :").grid(row=2, column=0, padx=10, pady=5, sticky="e")
glucose = tk.Entry(root)
glucose.grid(row=2, column=1, padx=10, pady=5)
tk.Label(root, text="Chỉ số BMI : ").grid(row=3, column=0, padx=10, pady=5, sticky="e")
bmi = tk.Entry(root)
bmi.grid(row=3, column=1, padx=10, pady=5)
tk.Label(root, text="Gender:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
gender_var = tk.StringVar(value="Male")
tk.OptionMenu(root, gender_var, "Male", "Female").grid(row=4, column=1, padx=10, pady=5)
# Hypertension
tk.Label(root, text="Hypertension (Yes/No):").grid(row=5, column=0, padx=10, pady=5, sticky="e")
hypertension_var = tk.StringVar(value="No")
tk.OptionMenu(root, hypertension_var, "Yes", "No").grid(row=5, column=1, padx=10, pady=5)
# Work related stress
tk.Label(root, text="Work related :").grid(row=6, column=0, padx=10, pady=5, sticky="e")
work_var = tk.StringVar(value="")
tk.OptionMenu(root, work_var, "Private", "Self-employed", "Govt_job", "Never_worked").grid(row=6, column=1, padx=10,
                                                                                           pady=5)
# Urban residence
tk.Label(root, text="Residence_type :").grid(row=7, column=0, padx=10, pady=5, sticky="e")
urban_var = tk.StringVar(value="")
tk.OptionMenu(root, urban_var, "Urban", "Rural").grid(row=7, column=1, padx=10, pady=5)
# Smokes
tk.Label(root, text="smoking_status :").grid(row=8, column=0, padx=10, pady=5, sticky="e")
smokes_var = tk.StringVar(value="Unknown")
tk.OptionMenu(root, smokes_var, "Unknown", "never smoked", "formerly smoked", "smokes").grid(row=8, column=1, padx=10,
                                                                                             pady=5)
tk.Label(root, text="Hear_disease(Yes/No):").grid(row=9, column=0, padx=10, pady=5, sticky="e")
Heart_var = tk.StringVar(value="No")
tk.OptionMenu(root, Heart_var, "Yes", "No").grid(row=9, column=1, padx=10, pady=5)
# Button
tk.Button(root, text="Predict", command=predict, bg="lightblue").grid(row=10, column=0, columnspan=2, pady=20)
# Label
tk.Label(root, text="Stroke: ").grid(row=11, column=0, padx=10, pady=5, sticky="e")
stroke = tk.Entry(root)
stroke.grid(row=11, column=1, padx=10, pady=5)

root.mainloop()