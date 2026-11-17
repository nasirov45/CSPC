import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

os.makedirs("results",exist_ok=True)

# A2 - Load, clean, and explore
df=pd.read_csv("data/concrete.csv")

print("Duplicate rows:",df.duplicated().sum())
print("\nMissing values:")
print(df.isna().sum())

df=df.drop_duplicates().reset_index(drop=True)

print("\nDataset description:")
print(df.describe())

plt.figure()
plt.scatter(df["cement"],df["strength"])
plt.xlabel("Cement (kg/m3)")
plt.ylabel("Strength (MPa)")
plt.title("Strength vs Cement")
plt.tight_layout()
plt.savefig("results/strength_cement.png")
plt.close()

plt.figure()
plt.scatter(df["age"],df["strength"])
plt.xlabel("Age (days)")
plt.ylabel("Strength (MPa)")
plt.title("Strength vs Age")
plt.tight_layout()
plt.savefig("results/strength_age.png")
plt.close()

# Rebuild X and y
X=df.drop("strength",axis=1)
y=df["strength"]

# A3 - Train/test split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("\nTrain size:",len(X_train))
print("Test size:",len(X_test))

# A4 - Linear Regression
model=LinearRegression()
model.fit(X_train,y_train)

y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
test_rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
test_mae=mean_absolute_error(y_test,y_test_pred)

train_r2=r2_score(y_train,y_train_pred)
test_r2=r2_score(y_test,y_test_pred)

n=len(y_train)
p=X_train.shape[1]
adjusted_r2=1-(1-train_r2)*(n-1)/(n-p-1)

print("\nLinear Regression Equation:")
print("Intercept:",model.intercept_)
for name,coef in zip(X.columns,model.coef_):
    print(name,":",coef)

print("\nRegression Metrics:")
metrics=pd.DataFrame({
    "Train RMSE":[train_rmse],
    "Test RMSE":[test_rmse],
    "Test MAE":[test_mae],
    "Train R2":[train_r2],
    "Adjusted Train R2":[adjusted_r2],
    "Test R2":[test_r2]
})
print(metrics.to_string(index=False))

residuals=y_test-y_test_pred
plt.figure()
plt.scatter(X_test["age"],residuals)
plt.xlabel("Age (days)")
plt.ylabel("Residuals (MPa)")
plt.title("Residuals vs Age")
plt.xscale("log")
plt.axhline(y=0,color="red",linestyle="--")
plt.tight_layout()
plt.savefig("results/residuals_age.png")
plt.close()

# A5 - Polynomial Regression
results=[]
for degree in range(1,5):
    poly=make_pipeline(
        PolynomialFeatures(degree=degree,include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    poly.fit(X_train,y_train)

    y_train_pred=poly.predict(X_train)
    y_test_pred=poly.predict(X_test)

    p=poly[0].n_output_features_
    n=len(y_train)

    train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
    test_rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))

    train_r2=r2_score(y_train,y_train_pred)
    test_r2=r2_score(y_test,y_test_pred)

    adjusted_r2=1-(1-train_r2)*(n-1)/(n-p-1)

    results.append({
        "Degree":degree,
        "Features":p,
        "Train RMSE":train_rmse,
        "Test RMSE":test_rmse,
        "Train R2":train_r2,
        "Adjusted Train R2":adjusted_r2,
        "Test R2":test_r2
    })

print("\nPolynomial Regression Results:")
print(pd.DataFrame(results).round(4).to_string(index=False))

# A6 - Classification: KNN & Decision Tree
y_class=(df["strength"]>=40).astype(int)

Xc_train,Xc_test,c_train,c_test=train_test_split(
    X,y_class,test_size=0.2,random_state=42,stratify=y_class
)

print("\nClass proportions:")
print(y_class.value_counts(normalize=True))
print("\nTraining class proportions:")
print(c_train.value_counts(normalize=True))
print("\nTest class proportions:")
print(c_test.value_counts(normalize=True))

k_values=[1,3,5,7,15,25,51,101]
knn_results=[]

for k in k_values:
    knn=make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )
    knn.fit(Xc_train,c_train)

    train_acc=knn.score(Xc_train,c_train)
    test_acc=knn.score(Xc_test,c_test)

    knn_results.append({
        "K":k,
        "Train Accuracy":train_acc,
        "Test Accuracy":test_acc
    })

knn_results=pd.DataFrame(knn_results)
print("\nKNN Results:")
print(knn_results.round(4).to_string(index=False))

plt.figure()
plt.plot(k_values,knn_results["Train Accuracy"],marker="o",label="Train")
plt.plot(k_values,knn_results["Test Accuracy"],marker="o",label="Test")
plt.xscale("log")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.legend()
plt.tight_layout()
plt.savefig("results/knn_accuracy.png")
plt.close()

# Scaled vs unscaled KNN
knn_scaled=make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=7)
)
knn_unscaled=KNeighborsClassifier(n_neighbors=7)

knn_scaled.fit(Xc_train,c_train)
knn_unscaled.fit(Xc_train,c_train)

print("\nScaled KNN Accuracy:",knn_scaled.score(Xc_test,c_test))
print("Unscaled KNN Accuracy:",knn_unscaled.score(Xc_test,c_test))

# KNN on five different splits
accuracies=[]
for random_state in range(5):
    X_train_split,X_test_split,y_train_split,y_test_split=train_test_split(
        X,y_class,test_size=0.2,
        random_state=random_state,stratify=y_class
    )
    knn=make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=7)
    )
    knn.fit(X_train_split,y_train_split)
    accuracy=knn.score(X_test_split,y_test_split)
    accuracies.append(accuracy)
    print("Random state",random_state,"Accuracy:",round(accuracy,4))

print("\nMean accuracy:",np.mean(accuracies))
print("Standard deviation:",np.std(accuracies))
print("Accuracy range:",max(accuracies)-min(accuracies))

# Decision Tree sweep
depth_values=[1,2,3,4,5,6,8,None]
tree_results=[]

for depth in depth_values:
    tree=DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        min_samples_leaf=1,
        random_state=0
    )
    tree.fit(Xc_train,c_train)

    train_acc=tree.score(Xc_train,c_train)
    test_acc=tree.score(Xc_test,c_test)
    leaves=tree.get_n_leaves()

    tree_results.append({
        "Depth":depth,
        "Leaves":leaves,
        "Train Accuracy":train_acc,
        "Test Accuracy":test_acc
    })

tree_results=pd.DataFrame(tree_results)
print("\nDecision Tree Results:")
print(tree_results.round(4).to_string(index=False))

plot_depths=[1,2,3,4,5,6,8,10]
plt.figure()
plt.plot(plot_depths,tree_results["Train Accuracy"],marker="o",label="Train")
plt.plot(plot_depths,tree_results["Test Accuracy"],marker="o",label="Test")
plt.xticks(plot_depths,["1","2","3","4","5","6","8","Unlimited"])
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs Depth")
plt.legend()
plt.tight_layout()
plt.savefig("results/tree_accuracy.png")
plt.close()

best_tree=DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    min_samples_leaf=1,
    random_state=0
)
best_tree.fit(Xc_train,c_train)

print("\nDecision Tree Structure:")
print(export_text(best_tree,feature_names=list(X.columns),max_depth=2))

# A7 - Classification Evaluation
models={
    "KNN":knn_scaled,
    "Decision Tree":best_tree
}

for name,model in models.items():
    predictions=model.predict(Xc_test)
    tn,fp,fn,tp=confusion_matrix(c_test,predictions).ravel()

    print("\n",name)
    print("TN:",tn)
    print("FP:",fp)
    print("FN:",fn)
    print("TP:",tp)
    print("Accuracy:",accuracy_score(c_test,predictions))
    print("Precision:",precision_score(c_test,predictions))
    print("Recall:",recall_score(c_test,predictions))
    print("F1:",f1_score(c_test,predictions))

# Threshold analysis using KNN
probabilities=knn_scaled.predict_proba(Xc_test)[:,1]

for threshold in [0.3,0.5,0.7,0.9]:
    predictions=(probabilities>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(c_test,predictions).ravel()

    print("\nThreshold:",threshold)
    print("TN:",tn,"FP:",fp,"FN:",fn,"TP:",tp)
    print("Precision:",precision_score(c_test,predictions))
    print("Recall:",recall_score(c_test,predictions))

# Save A7 metrics
metrics=[]
for name,model in models.items():
    predictions=model.predict(Xc_test)
    metrics.append({
        "Model":name,
        "Accuracy":accuracy_score(c_test,predictions),
        "Precision":precision_score(c_test,predictions),
        "Recall":recall_score(c_test,predictions),
        "F1":f1_score(c_test,predictions)
    })

metrics_df=pd.DataFrame(metrics)
metrics_df.to_csv("results/A_metrics.csv",index=False)

print("\nSaved A7 metrics:")
print(metrics_df.round(4).to_string(index=False))

# A8 - Feature Importance
linear_raw=LinearRegression()
linear_raw.fit(X_train,y_train)
raw_coefficients=linear_raw.coef_

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)

linear_standard=LinearRegression()
linear_standard.fit(X_train_scaled,y_train)
standard_coefficients=linear_standard.coef_

tree_importance=best_tree.feature_importances_

knn_permutation=permutation_importance(
    knn_scaled,Xc_test,c_test,
    scoring="f1",
    n_repeats=20,
    random_state=0
)

tree_permutation=permutation_importance(
    best_tree,Xc_test,c_test,
    scoring="f1",
    n_repeats=20,
    random_state=0
)

importance_df=pd.DataFrame({
    "Raw Coefficient":raw_coefficients,
    "Standardized Coefficient":standard_coefficients,
    "Tree Importance":tree_importance,
    "KNN Permutation":knn_permutation.importances_mean,
    "Tree Permutation":tree_permutation.importances_mean
},index=X.columns)

print("\nFeature Importance:")
print(importance_df.round(4).to_string())

importance_df.to_csv("results/A8_feature_importance.csv")

# Chart 1: Raw and standardized coefficient magnitudes
importance_df[["Raw Coefficient","Standardized Coefficient"]].abs().plot(
    kind="bar",figsize=(12,6)
)
plt.title("Raw vs Standardized Coefficients")
plt.ylabel("Absolute Coefficient")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/A8_coefficients.png")
plt.close()

# Chart 2: Decision Tree importance
importance_df["Tree Importance"].sort_values().plot(
    kind="bar",figsize=(10,6)
)
plt.title("Decision Tree Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("results/A8_tree_importance.png")
plt.close()

# Chart 3: Permutation importance
importance_df[["KNN Permutation","Tree Permutation"]].plot(
    kind="bar",figsize=(12,6)
)
plt.title("Permutation Feature Importance")
plt.ylabel("Mean F1 Score Decrease")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/A8_permutation_importance.png")
plt.close()