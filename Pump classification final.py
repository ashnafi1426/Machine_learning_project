# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
 
# FIX 1: display() only exists inside Jupyter/Colab. This makes it work in a normal .py file too.
try:
    from IPython.display import display
except ImportError:
    display = print
 
import warnings
warnings.filterwarnings("ignore")
 
 
# ============================================================
# 2. LOAD DATASET WITH COLUMN NAMES
# ============================================================
 
data = [
    [1, 1.2, 1500, 45, 8.5, 120, "Normal"],
    [2, 1.5, 1480, 48, 8.3, 250, "Normal"],
    [3, 1.8, 1520, 50, 8.6, 300, "Normal"],
    [4, 1.3, 1490, 46, 8.4, 180, "Normal"],
    [5, 2.0, 1510, 52, 8.2, 400, "Normal"],
    [6, 1.6, 1500, 49, 8.5, 220, "Normal"],
    [7, 2.1, 1470, 53, 8.1, 500, "Normal"],
    [8, 1.4, 1530, 47, 8.7, 150, "Normal"],
    [9, 2.2, 1490, 54, 8.3, 600, "Normal"],
    [10, 1.7, 1510, 50, 8.5, 350, "Normal"],
    [11, 2.5, 1450, 58, 7.9, 800, "Warning"],
    [12, 2.8, 1460, 60, 7.8, 950, "Warning"],
    [13, 3.0, 1440, 62, 7.7, 1100, "Warning"],
    [14, 3.2, 1430, 65, 7.5, 1250, "Warning"],
    [15, 2.7, 1470, 59, 7.9, 900, "Warning"],
    [16, 3.5, 1420, 67, 7.4, 1400, "Warning"],
    [17, 3.1, 1450, 63, 7.6, 1300, "Warning"],
    [18, 3.8, 1410, 69, 7.3, 1550, "Warning"],
    [19, 2.9, 1460, 61, 7.8, 1000, "Warning"],
    [20, 3.6, 1430, 68, 7.4, 1500, "Warning"],
    [21, 5.8, 1380, 78, 6.8, 2200, "Fault"],
    [22, 6.2, 1360, 82, 6.5, 2500, "Fault"],
    [23, 6.5, 1350, 85, 6.3, 2800, "Fault"],
    [24, 7.0, 1320, 89, 6.0, 3200, "Fault"],
    [25, 6.8, 1340, 87, 6.2, 3000, "Fault"],
    [26, 7.5, 1300, 92, 5.8, 3500, "Fault"],
    [27, 8.0, 1280, 95, 5.5, 3800, "Fault"],
    [28, 7.2, 1310, 90, 5.9, 3300, "Fault"],
    [29, 8.5, 1250, 98, 5.2, 4200, "Fault"],
    [30, 6.0, 1370, 80, 6.6, 2400, "Fault"],
    [31, 1.9, 1500, 51, 8.4, 380, "Normal"],
    [32, 2.3, 1480, 55, 8.0, 650, "Normal"],
    [33, 3.3, 1440, 64, 7.6, 1350, "Warning"],
    [34, 3.7, 1420, 70, 7.2, 1600, "Warning"],
    [35, 5.5, 1390, 76, 6.9, 2100, "Fault"],
    [36, 6.4, 1350, 84, 6.4, 2700, "Fault"],
    [37, 2.6, 1460, 57, 8.0, 850, "Warning"],
    [38, 1.8, 1520, 49, 8.6, 280, "Normal"],
    [39, 4.0, 1400, 72, 7.1, 1750, "Warning"],
    [40, 7.8, 1290, 94, 5.6, 3900, "Fault"]
]
 
# FIX 2: the column names did not match the order of the values in each row.
# Each row is: ID, vibration, RPM, temperature, pressure, operating hours, label.
# (Before, vibration values were being read as "Temperature_C", RPM as "Vibration", etc.)
columns = ["ID", "Vibration_mm_s", "Speed_RPM", "Temperature_C", "Pressure_bar", "Operating_Hours", "Failure_Risk"]
df = pd.DataFrame(data, columns=columns)
 
# Map status classes into binary target classes (0 = Normal, 1 = High Risk/Warning/Fault)
df["Failure_Risk"] = df["Failure_Risk"].map({"Normal": 0, "Warning": 1, "Fault": 1})
 
# Required columns
required_columns = [
    "ID",
    "Temperature_C",
    "Vibration_mm_s",
    "Speed_RPM",
    "Operating_Hours",
    "Failure_Risk"
]
 
# Validate dataset structure
missing_columns = set(required_columns) - set(df.columns)
 
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")
 
df = df[required_columns].copy()
 
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
 
display(df.head())
 
 
# ============================================================
# 4. FEATURE SELECTION
# ============================================================
 
# Input features
X = df[
    [
        "Temperature_C",
        "Vibration_mm_s",
        "Speed_RPM",
        "Operating_Hours"
    ]
]
 
# Target variable
y = df["Failure_Risk"]
 
# Check target classes
if y.isnull().any():
    raise ValueError("Target contains missing values.")
 
if not set(y.unique()).issubset({0, 1}) or len(y.unique()) != 2:
    raise ValueError("Expected two target classes: 0 and 1.")
 
print("Input features:")
print(X.columns.tolist())
 
# FIX 3: strings were split across two lines (SyntaxError). Use \n instead.
print("\nTarget variable:")
print("Failure_Risk")
 
print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
 
 
# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
 
print("TRAINING DATA")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
 
print("\nTESTING DATA")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
 
 
# ============================================================
# 6. DEFINE SIX CLASSIFICATION ALGORITHMS
# ============================================================
 
models = {
 
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=3))
    ]),
 
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ]),
 
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),
 
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),
 
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    ),
 
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
}
 
print("Six classification models created successfully!")
 
 
# ============================================================
# 7. MODEL TRAINING
# ============================================================
 
for name, model in models.items():
 
    model.fit(X_train, y_train)
 
    print(f"{name} trained successfully!")
 
print("\nAll six models have been trained.")
 

