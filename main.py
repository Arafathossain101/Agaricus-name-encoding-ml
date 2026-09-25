import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Column names from UCI Mushroom Dataset
columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

# Load dataset
df = pd.read_csv("agaricus-lepiota.data", header=None, names=columns)

# Step 1: Remove missing values
df = df[df["stalk-root"] != "?"]

# Step 2: Split into features and target
X = df.drop("class", axis=1)
y = df["class"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Encode features (train separately)
label_encoders = {}
for col in X_train.columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Encode target
le_target = LabelEncoder()
y_train = le_target.fit_transform(y_train)
y_test = le_target.transform(y_test)
label_encoders["class"] = le_target

# Step 5: Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc * 100
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# Step 6: Plot accuracy comparison
sns.set(style="whitegrid")
plt.figure(figsize=(10,5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="pastel")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(90, 100)
plt.tight_layout()
plt.savefig("model_accuracy.png")
plt.show()

# Step 7: Save best model and encoders (Random Forest here)

joblib.dump(models["Random Forest"], "best_model.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")
