# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import re
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack
import joblib

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("malicious_phish.csv")  # 👈 change filename

print("\nClass Distribution:\n", df['type'].value_counts())

# ==============================
# FEATURE ENGINEERING
# ==============================

def extract_features(url):
    return {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': len(re.findall(r'[^a-zA-Z0-9]', url)),
        'num_subdirs': url.count('/'),
        'num_hyphens': url.count('-'),
        'num_dots': url.count('.'),
        'has_https': int("https" in url),
        'has_ip': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        'has_at': int('@' in url),
        'has_double_slash': int(url.count('//') > 1),
        'has_suspicious_words': int(
            any(word in url.lower() for word in ['login', 'verify', 'bank', 'update', 'secure'])
        )
    }

print("\nExtracting features...")
features_df = df['url'].apply(extract_features).apply(pd.Series)

# ==============================
# TF-IDF FEATURES
# ==============================

print("\nGenerating TF-IDF features...")
tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=5000
)

X_tfidf = tfidf.fit_transform(df['url'])

# ==============================
# COMBINE FEATURES
# ==============================

scaler = StandardScaler()
X_numeric = scaler.fit_transform(features_df)

X = hstack([X_tfidf, X_numeric])

# ==============================
# LABEL ENCODING
# ==============================

le = LabelEncoder()
y = le.fit_transform(df['type'])

# ==============================
# HANDLE IMBALANCE
# ==============================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights_dict = dict(enumerate(class_weights))

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# OPTUNA TUNING
# ==============================

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
        "max_depth": trial.suggest_int("max_depth", 10, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": class_weights_dict
    }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return f1_score(y_test, preds, average='weighted')

print("\nRunning Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("\nBest Params:", study.best_params)
print("Best F1 Score:", study.best_value)

# ==============================
# FINAL MODEL
# ==============================

best_model = RandomForestClassifier(
    **study.best_params,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================

y_pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ==============================
# SAVE MODEL
# ==============================

joblib.dump(best_model, "rf_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel saved successfully!")