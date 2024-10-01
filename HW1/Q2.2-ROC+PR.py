import openml
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = openml.datasets.get_dataset(31, download_data=True, download_qualities=False, download_features_meta_data=False)
df = data.get_data()[0]

# Strip leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Convert the target 'class' to numeric values ('good' -> 1, 'bad' -> 0)
df['class'] = df['class'].map({'good': 1, 'bad': 0})

# Separate the target 'class' from features
y = df['class']
X = df.drop(columns='class')

# Convert categorical features to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Splitting Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Adaboost
clf1 = AdaBoostClassifier()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)

# Calculate ROC curve and PR curve
fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, y_pred, pos_label=1)
precision1, recall1, threshold2 = precision_recall_curve(y_test, y_pred, pos_label=1)

# Logistic Regression
clf2 = LogisticRegression()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

# Calculate ROC curve and PR curve for Logistic Regression
fpr2, tpr2, threshold3 = metrics.roc_curve(y_test, y_pred2, pos_label=1)
precision2, recall2, threshold4 = precision_recall_curve(y_test, y_pred2, pos_label=1)

# Plot Metrics for ROC
plt.figure(figsize=(8,6))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr1, tpr1, label='AdaBoost')
plt.plot(fpr2, tpr2, label='Logistic Regression')
plt.legend()
plt.grid()

# Plot Metrics for PR
plt.figure(figsize=(8,6))
plt.title('PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall1, precision1, label='AdaBoost')
plt.plot(recall2, precision2, label='Logistic Regression')
plt.legend()
plt.grid()

plt.show()