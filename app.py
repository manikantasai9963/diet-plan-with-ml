"""
NutriGuide AI – Flask Backend
==============================
Serves the trained RandomForest model via a REST API.

Endpoints
---------
GET  /api/health                → server & model status
GET  /api/model-info            → accuracy, feature importances, label distribution
POST /api/classify              → classify a food by nutrition values
POST /api/can-i-eat             → full "Can I eat this?" response
GET  /api/foods                 → paginated food list (with optional ?q= search)
GET  /api/recommend             → personalised recommendations
POST /api/meal-analyse          → analyse a full meal's nutrition

Run
---
    pip install flask flask-cors joblib scikit-learn pandas
    python app.py

For production use Gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
MODEL_PATH   = BASE / "model.pkl"
META_PATH    = BASE / "model_meta.json"
FOOD_DB_PATH = BASE / "food_db.json"

# ── App ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow the frontend (any origin) to call the API

# ── Load model & data ──────────────────────────────────────────────────────
print("Loading ML model …", end=" ", flush=True)
clf = joblib.load(MODEL_PATH)
CLASSES = clf.classes_.tolist()        # ['Healthy', 'Moderate', 'Unhealthy']
FEATURES = ["calories", "fat", "sugar", "sodium", "protein"]
print("OK")

with open(META_PATH) as f:
    MODEL_META = json.load(f)

with open(FOOD_DB_PATH) as f:
    FOOD_DB: List[Dict] = json.load(f)

# Build lookup dict  name_lower → record
FOOD_LOOKUP: Dict[str, Dict] = {item["name"].lower(): item for item in FOOD_DB}

print(f"  Model accuracy : {MODEL_META['accuracy_pct']}%")
print(f"  Food DB        : {len(FOOD_DB):,} items")


# ── Helper ─────────────────────────────────────────────────────────────────
def _feature_vec(data: Dict) -> np.ndarray:
    """Extract feature vector from a dict of nutrition values."""
    return np.array([[
        float(data.get("calories", data.get("cal", 0))),
        float(data.get("fat", 0)),
        float(data.get("sugar", 0)),
        float(data.get("sodium", 0)),
        float(data.get("protein", 0)),
    ]])


def _classify(vec: np.ndarray) -> Dict:
    """Run model.predict + predict_proba and return structured result."""
    label = clf.predict(vec)[0]
    probs = clf.predict_proba(vec)[0]
    prob_map = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}
    confidence = round(float(max(probs)), 4)
    return {"label": label, "confidence": confidence, "probabilities": prob_map}


def _get_reasons(food: Dict) -> List[Dict]:
    """Return human-readable reason tags based on nutrient values."""
    reasons = []
    cal, fat, sugar, sodium, protein = (
        food.get("cal", food.get("calories", 0)),
        food.get("fat", 0), food.get("sugar", 0),
        food.get("sodium", 0), food.get("protein", 0),
    )
    if cal > 450:
        reasons.append({"text": "High Calories",    "good": False})
    elif cal < 150 and cal > 0:
        reasons.append({"text": "Low Calories",     "good": True})

    if fat > 20:
        reasons.append({"text": "High Fat",         "good": False})
    elif fat < 5:
        reasons.append({"text": "Low Fat",          "good": True})

    if sugar > 15:
        reasons.append({"text": "High Sugar",       "good": False})
    elif sugar < 3:
        reasons.append({"text": "Low Sugar",        "good": True})

    if sodium > 600:
        reasons.append({"text": "High Sodium",      "good": False})
    elif sodium < 100:
        reasons.append({"text": "Low Sodium",       "good": True})

    if protein >= 15:
        reasons.append({"text": "High Protein ✓",   "good": True})
    elif protein < 3:
        reasons.append({"text": "Low Protein",      "good": False})

    return reasons


def _find_food(query: str) -> Optional[Dict]:
    """Fuzzy-ish lookup: exact → prefix → substring."""
    q = query.lower().strip()
    if q in FOOD_LOOKUP:
        return FOOD_LOOKUP[q]
    for name, item in FOOD_LOOKUP.items():
        if name.startswith(q):
            return item
    for name, item in FOOD_LOOKUP.items():
        if q in name:
            return item
    return None


def _healthy_alternatives(exclude_name: str, cal_limit: float = 300) -> List[Dict]:
    """Return up to 4 healthy foods under cal_limit."""
    alts = [
        item for item in FOOD_DB
        if item.get("mlLabel") == "Healthy"
        and item["cal"] > 30
        and item["cal"] < cal_limit
        and item["name"].lower() != exclude_name.lower()
    ]
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(alts), size=min(4, len(alts)), replace=False)
    return [alts[i] for i in indices]


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True,
                    "accuracy_pct": MODEL_META["accuracy_pct"],
                    "food_db_size": len(FOOD_DB)})


@app.route("/api/model-info")
def model_info():
    return jsonify(MODEL_META)


@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Body: { calories, fat, sugar, sodium, protein }
    Returns: { label, confidence, probabilities }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    vec = _feature_vec(data)
    result = _classify(vec)
    return jsonify(result)


@app.route("/api/can-i-eat", methods=["POST"])
def can_i_eat():
    """
    Body: { query: "food name", allergens: ["gluten", ...] }
    Returns full verdict card including reasons, alternatives, nutrient data.
    """
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    user_allergens = [a.lower() for a in data.get("allergens", [])]

    if not query:
        return jsonify({"error": "query is required"}), 400

    food = _find_food(query)
    if not food:
        return jsonify({"found": False, "query": query}), 404

    # Re-run model.predict (live inference, not stored label)
    vec = _feature_vec(food)
    ml_result = _classify(vec)
    reasons = _get_reasons(food)
    alts = _healthy_alternatives(food["name"])

    # Allergy check
    food_allergens = food.get("allergens", [])
    allergy_hit = [a for a in food_allergens if a.lower() in user_allergens]

    return jsonify({
        "found":        True,
        "food":         food,
        "verdict":      ml_result["label"],
        "confidence":   ml_result["confidence"],
        "probabilities": ml_result["probabilities"],
        "reasons":      reasons,
        "allergyWarning": allergy_hit,
        "alternatives": [{"name": a["name"], "cal": a["cal"],
                          "label": a.get("mlLabel", "Healthy")} for a in alts],
        "model":        "RandomForestClassifier",
        "n_estimators": 200,
    })


@app.route("/api/foods")
def list_foods():
    """
    ?q=search_term&label=Healthy&limit=20&offset=0
    Returns paginated food list.
    """
    q      = request.args.get("q", "").lower().strip()
    label  = request.args.get("label", "").strip()
    limit  = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))

    results = FOOD_DB
    if q:
        results = [item for item in results if q in item["name"].lower()]
    if label:
        results = [item for item in results if item.get("mlLabel") == label]

    total = len(results)
    page  = results[offset: offset + limit]
    return jsonify({"total": total, "limit": limit, "offset": offset, "items": page})


@app.route("/api/recommend")
def recommend():
    """
    ?calories=2000&diet=veg&goal=weight_loss&limit=8
    Returns ML-filtered personalised recommendations.
    """
    target_cal = float(request.args.get("calories", 2000))
    diet       = request.args.get("diet", "").lower()
    goal       = request.args.get("goal", "balanced").lower()
    limit      = int(request.args.get("limit", 8))

    # Per-meal calorie ceiling ≈ 30 % of daily
    meal_cal_limit = target_cal * 0.30

    candidates = [item for item in FOOD_DB if item["cal"] < meal_cal_limit and item["cal"] > 30]

    if goal in ("weight_loss", "fat_loss"):
        # Prefer Healthy + lower calorie
        candidates.sort(key=lambda x: (x.get("mlLabel") != "Healthy", x["cal"]))
    elif goal == "muscle_gain":
        # Prefer high protein
        candidates.sort(key=lambda x: (x.get("mlLabel") == "Unhealthy", -x["protein"]))
    else:
        candidates.sort(key=lambda x: (x.get("mlLabel") != "Healthy", x["cal"]))

    # Sample with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=7)
    top = candidates[:min(50, len(candidates))]
    pick_n = min(limit, len(top))
    indices = rng.choice(len(top), size=pick_n, replace=False)
    picks = [top[i] for i in sorted(indices)]

    return jsonify({
        "count":        len(picks),
        "goal":         goal,
        "target_cal":   target_cal,
        "recommendations": picks,
    })


@app.route("/api/meal-analyse", methods=["POST"])
def meal_analyse():
    """
    Body: { items: [ {name, cal, fat, sugar, sodium, protein}, … ] }
    Returns aggregated totals + per-item ML labels.
    """
    data = request.get_json(force=True)
    items = data.get("items", [])
    if not items:
        return jsonify({"error": "No items provided"}), 400

    enriched = []
    totals = {"cal": 0, "fat": 0, "sugar": 0, "sodium": 0, "protein": 0}

    for item in items:
        vec = _feature_vec(item)
        ml = _classify(vec)
        enriched.append({**item, "verdict": ml["label"], "confidence": ml["confidence"]})
        for k in totals:
            totals[k] += float(item.get(k, 0))

    # Classify the total meal
    meal_vec = _feature_vec(totals)
    meal_ml = _classify(meal_vec)

    return jsonify({
        "items":        enriched,
        "totals":       {k: round(v, 1) for k, v in totals.items()},
        "meal_verdict": meal_ml["label"],
        "meal_confidence": meal_ml["confidence"],
    })


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nNutriGuide AI backend running → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
