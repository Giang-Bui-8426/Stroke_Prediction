import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as Pipeline2
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as Pipeline1
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif

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
    y_pred = best_model.predict(data_predict)

    # Nếu có dùng LabelEncoder cho y, giải mã lại
    pred1 = "Yes" if encoders["stroke"].inverse_transform(y_pred)[0] == 1 else "No"
    stroke.delete(0, tk.END)
    stroke.insert(0,pred1)

def clean_missData(data,col_data):
    for i in  col_data:
        if i not in data:
            data[i] = np.nan
    return data[col_data]
data = pd.read_csv("stroke_data.csv")
data = data.drop("id",axis=1)
data["bmi"] = (data["bmi"].fillna(data["bmi"].median()))
data.loc[~data["gender"].isin(["Male", "Female"]), "gender"] = data["gender"].mode()[0]
data["age"] = data["age"].apply(lambda x: int(round(x)) if x > 1 else 1)
# file = ProfileReport(data,title = "report_stroke")
# file.to_file("report_stroke.html")
x = data.drop(["stroke","ever_married"],axis=1)
y = data["stroke"]

encoders = {}
label = LabelEncoder()
y = pd.Series(label.fit_transform(y))
encoders["stroke"] = label

nominal_cols = x.columns.difference(["age","bmi","avg_glucose_level"]).tolist() # lấy column trừ Age
encodAndStand = Pipeline1(steps=[("simple",SimpleImputer(strategy="most_frequent")),
                        ("encoder",OneHotEncoder(handle_unknown="ignore"))]) # chuẩn hóa các value về dạng số
stand = Pipeline1(steps=[("simple",SimpleImputer(strategy='mean')), # chuẩn hóa dể tránh model bị ảnh hưởng bởi outlier
                        ("stand", StandardScaler())])
processor = ColumnTransformer(transformers=[("encod_feature" ,encodAndStand,nominal_cols),
                                            ("stand_feature",stand,["age","bmi","avg_glucose_level"])])

models = {"Support Vector Machine":SVC(random_state=1,class_weight="balanced"),
        "DecisionTree" : DecisionTreeClassifier(random_state=1,class_weight="balanced"),
          "RandomForest" : RandomForestClassifier(n_estimators=200,max_depth=10,random_state=1,class_weight="balanced")}
save_result_model = {}
x = processor.fit_transform(x)

ohe = processor.named_transformers_["encod_feature"].named_steps["encoder"]
encoded_cols = ohe.get_feature_names_out(processor.transformers_[0][2])
num_cols = processor.transformers_[1][2]
all_cols = list(encoded_cols) + num_cols

x= pd.DataFrame(x, columns=all_cols)
smote_tomek = SMOTETomek(sampling_strategy=1,random_state=1)
x, y = smote_tomek.fit_resample(x, y)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
for name,model in models.items():
    scores_precision_0 = []
    scores_precision_1 = []
    scores_recall_0 = []
    scores_recall_1 = []
    scores_accuracy = []
    scores_f1_0 = []
    scores_f1_1 =  []
    reg = model
    if name == "RandomForest": # khởi tạo Gridsearch cho random
        param = {"n_estimators": [200, 300],"max_features": ["sqrt"],
        "min_samples_split": [5,7],
        "criterion": ["gini", "entropy"]}
        reg = GridSearchCV(estimator=RandomForestClassifier(class_weight="balanced", random_state=1),param_grid=param,cv=4,scoring="recall",verbose=2) # giúp thay đổi parameter của random giúp đa dạng hơn
    for train, test in kfold.split(x,y):
        reg.fit(x.iloc[train, :], y.iloc[train].values.ravel())

        predict = reg.predict(x.iloc[test, :])
        recall = recall_score(y.iloc[test], predict, average=None)
        precision = precision_score(y.iloc[test], predict, average=None)
        f1 = f1_score(y.iloc[test], predict,average=None)

        scores_accuracy.append(accuracy_score(y.iloc[test], predict))
        scores_precision_0.append(precision[0])
        scores_precision_1.append(precision[1])
        scores_recall_1.append(recall[1])
        scores_recall_0.append(recall[0])
        scores_f1_0.append(f1[0])
        scores_f1_1.append(f1[1])


    # save_result_model[name] = {"Stroke" : { "accuracy" : acc_stroke , "recall" : recall_stroke ,"f1" : f1_stroke ,"precision" : precision_stroke}}
    print(f"{name} : ")
    print(f"Accuracy trung bình: {np.mean(scores_accuracy)}")
    print(f"Precision trung bình: 0 : {np.mean(scores_precision_0)} || 1 : {np.mean(scores_precision_1)}")
    print(f"Recall trung bình: 0 : {np.mean(scores_recall_0)} || 1 : {np.mean(scores_recall_1)}")
    print(f"F1 trung bình: 0 : {np.mean(scores_f1_0)} || 1 : {np.mean(scores_f1_1)}")
best_model = reg.best_estimator_
col = x.columns
root = tk.Tk()
root.title("Stroke Prediction")
root.geometry("400x500")
tk.Label(root, text="Age:").grid(row=1, column=0, padx=10, pady=5, sticky="e",)
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
tk.OptionMenu(root, work_var, "Private", "Self-employed","Govt_job","Never_worked").grid(row=6, column=1, padx=10, pady=5)
# Urban residence
tk.Label(root, text="Residence_type :").grid(row=7, column=0, padx=10, pady=5, sticky="e")
urban_var = tk.StringVar(value="")
tk.OptionMenu(root, urban_var, "Urban", "Rural").grid(row=7, column=1, padx=10, pady=5)
# Smokes
tk.Label(root, text="smoking_status :").grid(row=8, column=0, padx=10, pady=5, sticky="e")
smokes_var = tk.StringVar(value="Unknown")
tk.OptionMenu(root, smokes_var, "Unknown", "never smoked","formerly smoked","smokes").grid(row=8, column=1, padx=10, pady=5)
tk.Label(root, text="Hear_disease(Yes/No):").grid(row=9, column=0, padx=10, pady=5, sticky="e")
Heart_var = tk.StringVar(value="No")
tk.OptionMenu(root, Heart_var, "Yes", "No").grid(row=9, column=1, padx=10, pady=5)
# Button
tk.Button(root, text="Predict", command=predict, bg="lightblue").grid(row=10,column=0, columnspan=2, pady=20)
#Label
tk.Label(root, text="Stroke: ").grid(row=11, column=0, padx=10, pady=5, sticky="e")
stroke = tk.Entry(root)
stroke.grid(row=11, column=1, padx=10, pady=5)

root.mainloop()
