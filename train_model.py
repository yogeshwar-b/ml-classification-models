import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os


print("Loading Mobile Price Dataset...")
df = pd.read_csv('Dataset/mobileprice.csv')

target_col = 'Price_Range'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Full Dataset: {X.shape}")
print(f"Target Classes: {y.unique()}")

# Data Split 80-20
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data: {X_train_raw.shape}")
print(f"Validation Data: {X_test_raw.shape}")

# PreProcessing
scaler = StandardScaler()

print("Scaling features...")
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_names.pkl')

print("Preprocessing complete.")

# Add Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

# Training
results = {}

print("\nTraining Models...")
for name, model in models.items():
    print(f"Training {name}...")
    
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)
    
    # Accuracy
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.5
        
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, preds)
    
    results[name] = [acc, auc, prec, rec, f1, mcc]
    
    # Save Model
    joblib.dump(model, f"model/{name.replace(' ','_').lower()}.pkl")

#Results
print("\n" + "="*80)
print("Results")
print("="*80)
print(f"{'Model':<20} | {'Acc':<6} | {'AUC':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'MCC':<6}")
print("-" * 80)
for k, v in results.items():
    print(f"{k:<20} | {v[0]:.3f}  | {v[1]:.3f}  | {v[2]:.3f}  | {v[3]:.3f}  | {v[4]:.3f}  | {v[5]:.3f}")
print("-" * 80)