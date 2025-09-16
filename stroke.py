import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek
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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import joblib

# Sử dụng hàm loss chi phí để chọn ngưỡng do mô hình bị lệch quá cao mặc dù các chỉ số khi test cao nhưng phần lớn là dữ liệu ảo
def cost_loss(y_true, y_proba, threshold, C_FN=10, C_FP=1):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    loss = C_FN * fn + C_FP * fp
    return loss
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
best_model = ""
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
    if name == "RandomForest":
        best_model = reg.best_estimator_

        y_proba = best_model.predict_proba(x.iloc[test, :])[:, 1] # lấy xác suất của cây có parameter tốt nhất
        thresholds = np.linspace(0, 1, 100)
        losses = [cost_loss(y.iloc[test], y_proba, t, C_FN=10, C_FP=1) for t in thresholds]
        best_threshold = thresholds[np.argmin(losses)]
        print("Best threshold:", best_threshold)
        joblib.dump(best_threshold, "threshold.pkl")
        # save_result_model[name] = {"Stroke" : { "accuracy" : acc_stroke , "recall" : recall_stroke ,"f1" : f1_stroke ,"precision" : precision_stroke}}
    print(f"{name} : ")
    print(f"Accuracy trung bình: {np.mean(scores_accuracy)}")
    print(f"Precision trung bình: 0 : {np.mean(scores_precision_0)} || 1 : {np.mean(scores_precision_1)}")
    print(f"Recall trung bình: 0 : {np.mean(scores_recall_0)} || 1 : {np.mean(scores_recall_1)}")
    print(f"F1 trung bình: 0 : {np.mean(scores_f1_0)} || 1 : {np.mean(scores_f1_1)}")
joblib.dump(best_model, "stroke_model.pkl")
joblib.dump(processor, "processor.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(all_cols, "all_cols.pkl")

