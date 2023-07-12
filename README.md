# Dry_Bean-Project-2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Dataset
data = pd.read_csv('dry-beans.csv')

# Data Cleaning
data.isnull().sum()

# Exploratory Data Analysis (EDA)
data.head()
data.info()
data.describe()
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Data Preprocessing
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

# Compare the Performances
accuracy_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'kNN', 'Naïve Bayes', 'SVM'],
    'Accuracy': [accuracy_lr, accuracy_dt, accuracy_knn, accuracy_nb, accuracy_svm]
})
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

# Print the results
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Report:")
print(report_lr)
print("-------------------------------")
print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Report:")
print(report_dt)
print("-------------------------------")
print("kNN Accuracy:", accuracy_knn)
print("kNN Report:")
print(report_knn)
print("-------------------------------")
print("Naïve Bayes Accuracy:", accuracy_nb)
print("Naïve Bayes Report:")
print(report_nb)
print("-------------------------------")
print("SVM Accuracy:", accuracy_svm)
print("SVM Report:")
print(report_svm)
print("-------------------------------")
print(accuracy_df)

# Conclusion
print("CONCLUSION")
print("[Add your conclusion statement here]")

# Future Work
print("FUTURE WORK")
print("[Add future work or suggestions for improvement]")

