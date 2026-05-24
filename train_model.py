"""
NutriGuide AI – Random Forest Training Pipeline
================================================
Trains a RandomForestClassifier on 5 merged nutrition datasets.
Outputs:
  model.pkl        – serialised model (joblib)
  model_meta.json  – accuracy, feature importances, label distribution
  food_db.json     – full food DB with ML labels & confidence scores

Usage:
    python train_model.py
    python train_model.py --data-dir /path/to/csvs --out-dir /path/to/output
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train NutriGuide ML model")
parser.add_argument("--data-dir", default=".", help="Directory containing input CSVs")
parser.add_argument("--out-dir",  default=".", help="Directory for model outputs")
args, _ = parser.parse_known_args()

DATA_DIR = Path(args.data_dir)
OUT_DIR  = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["calories", "fat", "sugar", "sodium", "protein"]

# ── Label function (evidence-based nutrition thresholds) ───────────────────
def label_food(row: pd.Series) -> str:
    """
    Assigns Healthy / Moderate / Unhealthy based on a weighted scoring
    of five key nutrients per serving / per 100 g.
    """
    cal, fat, sugar, sodium, protein = (
        row["calories"], row["fat"], row["sugar"], row["sodium"], row["protein"]
    )
    score = 0
    # Calorie density
    if cal > 450:   score += 2
    elif cal > 250: score += 1
    # Fat
    if fat > 20:    score += 2
    elif fat > 10:  score += 1
    # Sugar
    if sugar > 15:  score += 2
    elif sugar > 8: score += 1
    # Sodium
    if sodium > 600:   score += 2
    elif sodium > 300: score += 1
    # Protein is beneficial
    if protein >= 15:  score -= 2
    elif protein >= 8: score -= 1

    if score <= 1:  return "Healthy"
    if score <= 3:  return "Moderate"
    return "Unhealthy"


def load_datasets(data_dir: Path) -> pd.DataFrame:
    """Merge all available CSV files into a single normalised DataFrame."""
    records = []

    # 1. Indian Food Nutrition Processed
    p = data_dir / "Indian_Food_Nutrition_Processed.csv"
    if p.exists():
        df = pd.read_csv(p).dropna()
        for _, r in df.iterrows():
            records.append({"name": r["Dish Name"],
                             "calories": r["Calories (kcal)"],
                             "fat":      r["Fats (g)"],
                             "sugar":    r["Free Sugar (g)"],
                             "sodium":   r["Sodium (mg)"],
                             "protein":  r["Protein (g)"]})
        print(f"  ✓ Indian_Food_Nutrition_Processed.csv  →  {len(df)} rows")

    # 2. Cleaned nutrition per 100 g
    p = data_dir / "cleaned_nutrition_dataset_per100g.csv"
    if p.exists():
        req = ["Calories (kcal per 100g)", "Fat (g per 100g)",
               "Sugars (g per 100g)", "Sodium (mg per 100g)", "Protein (g per 100g)"]
        df = pd.read_csv(p).dropna(subset=req)
        for _, r in df.iterrows():
            records.append({"name":     r["food"],
                             "calories": r["Calories (kcal per 100g)"],
                             "fat":      r["Fat (g per 100g)"],
                             "sugar":    r["Sugars (g per 100g)"],
                             "sodium":   r["Sodium (mg per 100g)"] * 1000,
                             "protein":  r["Protein (g per 100g)"]})
        print(f"  ✓ cleaned_nutrition_dataset_per100g.csv →  {len(df)} rows")

    # 3. Filtered meals (regional Indian)
    p = data_dir / "filtered_meals.csv"
    if p.exists():
        req = ["Total Calories", "Total Fats", "Total Sugar", "Total Sodium", "Total Protein"]
        df = pd.read_csv(p).dropna(subset=req)
        for _, r in df.iterrows():
            records.append({"name":     r["Food Name"],
                             "calories": r["Total Calories"],
                             "fat":      r["Total Fats"],
                             "sugar":    r["Total Sugar"],
                             "sodium":   r["Total Sodium"],
                             "protein":  r["Total Protein"]})
        print(f"  ✓ filtered_meals.csv  →  {len(df)} rows")

    # 4. USDA food.csv
    p = data_dir / "food.csv"
    if p.exists():
        req = ["Data.Kilocalories", "Data.Fat.Total Lipid",
               "Data.Sugar Total", "Data.Major Minerals.Sodium", "Data.Protein"]
        df = pd.read_csv(p).dropna(subset=req)
        for _, r in df.iterrows():
            records.append({"name":     r["Description"],
                             "calories": r["Data.Kilocalories"],
                             "fat":      r["Data.Fat.Total Lipid"],
                             "sugar":    r["Data.Sugar Total"],
                             "sodium":   r["Data.Major Minerals.Sodium"],
                             "protein":  r["Data.Protein"]})
        print(f"  ✓ food.csv  →  {len(df)} rows")

    # 5. test.csv
    p = data_dir / "test.csv"
    if p.exists():
        req = ["Energy_kcal", "Fat_g", "Sugar_g", "Protein_g"]
        df = pd.read_csv(p).dropna(subset=req)
        for _, r in df.iterrows():
            records.append({"name":     r["Descrip"],
                             "calories": r["Energy_kcal"],
                             "fat":      r["Fat_g"],
                             "sugar":    r["Sugar_g"],
                             "sodium":   0,
                             "protein":  r["Protein_g"]})
        print(f"  ✓ test.csv  →  {len(df)} rows")

    combined = pd.DataFrame(records).dropna()
    combined = combined[combined["calories"] > 0].reset_index(drop=True)

    # Cap outliers at 99th percentile to prevent skewed splits
    for col in FEATURES:
        cap = combined[col].quantile(0.99)
        combined[col] = combined[col].clip(upper=cap)

    return combined


def train(data_dir: Path, out_dir: Path) -> None:
    print("\n── Loading datasets ──────────────────────────────────────────────")
    df = load_datasets(data_dir)
    print(f"\n  Total records after merge & clean: {len(df):,}")

    # Generate labels
    df["label"] = df.apply(label_food, axis=1)
    print("\n── Label distribution ───────────────────────────────────────────")
    print(df["label"].value_counts().to_string())

    X = df[FEATURES].values
    y = df["label"].values

    # ── Train / Test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n── Splitting  →  train: {len(X_train):,}  |  test: {len(X_test):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n── Training RandomForestClassifier ──────────────────────────────")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  ✓ Test accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print("\n── Classification report ────────────────────────────────────────")
    print(classification_report(y_test, y_pred))

    fi = dict(zip(FEATURES, clf.feature_importances_))
    print("── Feature importances ──────────────────────────────────────────")
    for k, v in sorted(fi.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 40)
        print(f"  {k:12s}  {v:.4f}  {bar}")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = out_dir / "model.pkl"
    joblib.dump(clf, model_path)
    print(f"\n  ✓ Model saved  →  {model_path}")

    # ── Save metadata ─────────────────────────────────────────────────────
    meta = {
        "accuracy":            round(acc, 4),
        "accuracy_pct":        round(acc * 100, 2),
        "train_samples":       int(len(X_train)),
        "test_samples":        int(len(X_test)),
        "total_samples":       int(len(df)),
        "features":            FEATURES,
        "classes":             clf.classes_.tolist(),
        "n_estimators":        200,
        "feature_importances": {k: round(float(v), 4) for k, v in fi.items()},
        "label_distribution":  {k: int(v) for k, v in df["label"].value_counts().items()},
    }
    meta_path = out_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Metadata     →  {meta_path}")

    # ── Export food DB with ML labels ─────────────────────────────────────
    db = df[["name", "calories", "fat", "sugar", "sodium", "protein", "label"]].copy()
    db.columns = ["name", "cal", "fat", "sugar", "sodium", "protein", "mlLabel"]
    db = db.drop_duplicates(subset="name").reset_index(drop=True)

    probs = clf.predict_proba(db[FEATURES].rename(columns={"cal": "calories"}).values
                              if False else db[["cal", "fat", "sugar", "sodium", "protein"]].values)
    prob_df = pd.DataFrame(probs, columns=clf.classes_)
    db["confidence"] = prob_df.max(axis=1).round(3)

    food_list = db.to_dict(orient="records")
    for item in food_list:
        for k in ["cal", "fat", "sugar", "sodium", "protein", "confidence"]:
            item[k] = round(float(item[k]), 2)

    db_path = out_dir / "food_db.json"
    with open(db_path, "w") as f:
        json.dump(food_list, f)
    print(f"  ✓ Food DB      →  {db_path}  ({len(food_list):,} items)")
    print("\n── Done ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    train(DATA_DIR, OUT_DIR)
