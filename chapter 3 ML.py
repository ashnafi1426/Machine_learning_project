"""
Practical Engineering Case Study: Supervised Machine Learning Classification
Algorithms:
1. Support Vector Machines (SVM)
2. Naïve Bayes (Gaussian Naïve Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)
# ============================================================

data = [
    # ID, Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Failure_Risk
    [1,  53,  6.2, 1485,  905, 0],
    [2,  56,  6.5, 1495, 1205, 0],
    [3,  60,  6.8, 1480, 1505, 0],
    [4,  63,  7.0, 1505, 1805, 0],
    [5,  65,  7.2, 1465, 2105, 0],
    [6,  58,  6.4, 1515, 1105, 0],
    [7,  67,  7.4, 1455, 2305, 0],
    [8,  62,  6.7, 1500, 1605, 0],
    [9,  69,  7.6, 1445, 2505, 0],
    [10, 55,  6.3, 1510, 1005, 0],
    [11, 71,  7.8, 1440, 2705, 0],
    [12, 64,  7.1, 1490, 1905, 0],
    [13, 68,  7.5, 1470, 2205, 0],
    [14, 61,  6.9, 1503, 1405, 0],
    [15, 66,  7.3, 1475, 2005, 0],
    [16, 87, 10.8, 1375, 5205, 1],
    [17, 93, 11.5, 1345, 6105, 1],
    [18, 96, 12.1, 1325, 6805, 1],
    [19, 83, 10.2, 1395, 4805, 1],
    [20, 100, 12.8, 1295, 7405, 1],
    [21, 90, 11.1, 1365, 5705, 1],
    [22, 104, 13.4, 1265, 8105, 1],
    [23, 81,  9.9, 1405, 4505, 1],
    [24, 95, 12.0, 1315, 7005, 1],
    [25, 92, 11.7, 1335, 6405, 1],
    [26, 107, 13.8, 1245, 8605, 1],
    [27, 85, 10.5, 1385, 5005, 1],
    [28, 98, 12.5, 1285, 7605, 1],
    [29, 89, 11.0, 1355, 5905, 1],
    [30, 102, 13.1, 1275, 7905, 1],
]

columns = [
    "Machine_ID", "Temperature_C", "Vibration_mm_s",
    "Speed_RPM", "Operating_Hours", "Failure_Risk"
]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 MACHINE RECORDS")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Failure_Risk"].value_counts().rename(index={0: "Normal", 1: "Failure risk"}))

# Separate predictors (X) and target (y).
feature_cols = ["Temperature_C", "Vibration_mm_s", "Speed_RPM", "Operating_Hours"]
X = df[feature_cols]
y = df["Failure_Risk"]

# Initial data visualization.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Failure_Risk", style="Failure_Risk", s=100
)
plt.title("Machine temperature vs. vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Failure_Risk"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of engineering variables")
plt.tight_layout()
plt.show()

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

# Stratification preserves the class ratio in both subsets.
# With only 30 records, test metrics will be highly sensitive to individual records.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

# SVM: scaling is important because SVM uses distances/margins.
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# Gaussian Naive Bayes: assumes feature values are conditionally Gaussian
# and conditionally independent given the class.
naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

# Decision Tree: learns a sequence of feature-based splitting rules.
decision_tree_model = DecisionTreeClassifier(
    max_depth=3, random_state=42
)

# Random Forest: combines many decision trees trained on bootstrap samples.
random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)

# AdaBoost: sequentially trains weak learners, emphasizing previously
# misclassified training examples.
# SAMME is explicitly selected for compatibility across sklearn versions.
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    # Learn from the training records.
    model.fit(X_train, y_train)

    # Predict labels for unseen test records.
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Normal", "Failure risk"],
        zero_division=0
    ))

    # Confusion matrix visualization for this algorithm.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Failure risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of five supervised learning algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Engineering feature")
    plt.tight_layout()
    plt.show()

# Visualize the learned Decision Tree.
plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Failure risk"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree learned from the training records")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. TEST A NEW ENGINEERING OBSERVATION
# ============================================================

# Example values are illustrative. They are not a real machine diagnosis only for you(Software Engineering 4th year students).
new_machine = pd.DataFrame([{
    "Temperature_C": 94,
    "Vibration_mm_s": 11.8,
    "Speed_RPM": 1330,
    "Operating_Hours": 6505
}])

print("\nNEW MACHINE OBSERVATION")
print(new_machine)

for name, model in models.items():
    predicted_class = int(model.predict(new_machine)[0])
    label = "Failure risk" if predicted_class == 1 else "Normal"
    print(f"{name:15s} prediction: {label}")

print("\nEND OF PRACTICAL CASE STUDY")