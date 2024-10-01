import openml
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = openml.datasets.get_dataset(31, download_data=True, download_qualities=False, download_features_meta_data=False)
df = data.get_data()[0]

# Strip leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

print(df.head()) 

# Convert the target 'class' to numeric values ('good' -> 1, 'bad' -> 0)
df['class'] = df['class'].map({'good': 1, 'bad': 0})

# Check if 'class' is numeric now
print(df['class'].dtype)  # Should print int64 or similar

# Ensure 'class' is numeric (if needed, cast to int)
df['class'] = df['class'].astype(int)

# Separate the target 'class' from features
y = df['class']
X = df.drop(columns='class')

# Convert categorical features to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Splitting Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train AdaBoost Classifier
clf1 = AdaBoostClassifier()
clf1.fit(X_train, y_train)
y_pred_ada = clf1.predict(X_test)

# Train Logistic Regression
clf2 = LogisticRegression()
clf2.fit(X_train, y_train)
y_pred_lr = clf2.predict(X_test)

# 1. Calculate AUROC (Area under ROC Curve)
auroc_ada = roc_auc_score(y_test, y_pred_ada)
auroc_lr = roc_auc_score(y_test, y_pred_lr)

# 2. Calculate AUPR (Area under PR Curve)
precision_ada, recall_ada, _ = precision_recall_curve(y_test, y_pred_ada)
aupr_ada = auc(recall_ada, precision_ada)

precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_lr)
aupr_lr = auc(recall_lr, precision_lr)

# 3. Manual PR Gain curve calculation (Precision-Recall Gain Curve)
def calculate_prg_curve(precision, recall, prevalence):
    precision_gain = (precision - prevalence) / (1 - prevalence)
    recall_gain = recall / prevalence
    return precision_gain, recall_gain

# Calculate Prevalence (proportion of positives)
prevalence = np.sum(y_test) / len(y_test)

# PR Gain Curve for AdaBoost
precision_gain_ada, recall_gain_ada = calculate_prg_curve(precision_ada, recall_ada, prevalence)
auprg_ada = auc(recall_gain_ada, precision_gain_ada)

# PR Gain Curve for Logistic Regression
precision_gain_lr, recall_gain_lr = calculate_prg_curve(precision_lr, recall_lr, prevalence)
auprg_lr = auc(recall_gain_lr, precision_gain_lr)

# Print the results
print(f"AdaBoost AUROC: {auroc_ada:.3f}, AUPR: {aupr_ada:.3f}, AUPRG: {auprg_ada:.3f}")
print(f"Logistic Regression AUROC: {auroc_lr:.3f}, AUPR: {aupr_lr:.3f}, AUPRG: {auprg_lr:.3f}")

# Plot the ROC Curves
plt.figure(figsize=(8,6))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
fpr_ada, tpr_ada, _ = metrics.roc_curve(y_test, y_pred_ada, pos_label=1)
fpr_lr, tpr_lr, _ = metrics.roc_curve(y_test, y_pred_lr, pos_label=1)
plt.plot(fpr_ada, tpr_ada, label='AdaBoost')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.legend()
plt.grid()

# Plot the PR Curves
plt.figure(figsize=(8,6))
plt.title('PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall_ada, precision_ada, label='AdaBoost')
plt.plot(recall_lr, precision_lr, label='Logistic Regression')
plt.legend()
plt.grid()

# Plot the PR Gain Curves
plt.figure(figsize=(8,6))
plt.title('PR Gain Curve')
plt.xlabel('Recall Gain')
plt.ylabel('Precision Gain')
plt.plot(recall_gain_ada, precision_gain_ada, label='AdaBoost')
plt.plot(recall_gain_lr, precision_gain_lr, label='Logistic Regression')
plt.legend()
plt.grid()

plt.show()