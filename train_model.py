"""Train the decision tree (settings from the class code's grid search) and save it for the app."""
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_excel("delivery_delay.xlsx")
X, y = df.drop(columns="Delivery_Delay"), df["Delivery_Delay"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, random_state=42)
tree.fit(X_train, y_train)

pred = tree.predict(X_test)
print("Test accuracy:", round(tree.score(X_test, y_test), 3), "| depth:", tree.get_depth(), "| leaves:", tree.get_n_leaves())
print("Confusion matrix [[TN FP],[FN TP]]:", confusion_matrix(y_test, pred).tolist())
print(classification_report(y_test, pred))

joblib.dump({"tree": tree, "columns": list(X.columns)}, "delay_model.joblib")
print("saved")
