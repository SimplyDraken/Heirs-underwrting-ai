import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "past_cases.csv"

# -------- Helper: build a conservative "confidence" score ----------
def compute_confidence(data_completeness: float, similarity_score: float) -> int:
    """
    Confidence is not "how safe" the customer is.
    It is how reliable our estimate is.
    We combine:
      - how complete the input data is
      - how similar this case is to past cases (coverage)
    """
    # Weight completeness more heavily because it's easy to justify to judges
    raw = 0.7 * (data_completeness / 100.0) + 0.3 * similarity_score
    return int(round(100 * np.clip(raw, 0, 1)))


def recommend_action(risk_score: int, confidence_score: int) -> str:
    """
    Guardrails:
      - Low confidence -> mandatory human review
      - High risk -> limited coverage
      - Else -> eligible for standard approval
    """
    if confidence_score < 40:
        return "ESCALATE (human review required)"
    if risk_score > 70:
        return "LIMITED COVERAGE (higher deductible/short term)"
    return "APPROVE ELIGIBLE (standard terms)"


def main():
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df.drop(columns=["claim_outcome"])
    y = df["claim_outcome"].astype(int)

    # Define categorical and numeric columns
    cat_cols = ["occupation_risk", "location_risk", "income_stability", "prior_insurance"]
    num_cols = ["age", "data_completeness"]

    # Preprocessing: one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Model: simple, explainable baseline
    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    # Train (for prototype we train on all data; in real life you'd validate properly)
    clf.fit(X, y)

    # ---- New applicant example (edit these values to demo) ----
    new_applicant = {
        "age": 27,
        "occupation_risk": "med",
        "location_risk": "high",
        "income_stability": "med",
        "prior_insurance": "no",
        "data_completeness": 55
    }
    new_df = pd.DataFrame([new_applicant])

    # Risk score = probability of claim * 100
    prob_claim = float(clf.predict_proba(new_df)[0, 1])
    risk_score = int(round(prob_claim * 100))

    # Similarity: compare applicant to past cases in feature space
    X_transformed = clf.named_steps["prep"].transform(X)
    new_transformed = clf.named_steps["prep"].transform(new_df)
    sims = cosine_similarity(new_transformed, X_transformed)[0]
    top_idx = np.argsort(sims)[::-1][:3]
    top_sim = float(sims[top_idx[0]])

    confidence_score = compute_confidence(
        data_completeness=float(new_applicant["data_completeness"]),
        similarity_score=top_sim
    )

    action = recommend_action(risk_score, confidence_score)

    # Show top similar cases
    similar_cases = df.iloc[top_idx].copy()
    similar_cases["similarity"] = sims[top_idx].round(3)

    print("\n=== AI-Assisted Underwriting Prototype Output ===")
    print("Applicant:", new_applicant)
    print(f"\nRisk Score: {risk_score}/100 (estimated claim probability: {prob_claim:.2f})")
    print(f"Confidence Score: {confidence_score}/100 (data + similarity coverage)")
    print("Recommendation:", action)

    print("\nTop 3 Similar Past Cases (for explainability):")
    print(similar_cases[[
        "age", "occupation_risk", "location_risk", "income_stability",
        "prior_insurance", "data_completeness", "claim_outcome", "similarity"
    ]].to_string(index=False))

    print("\nNotes:")
    print("- Risk = likelihood of loss based on learned patterns from past outcomes.")
    print("- Confidence = how reliable the estimate is when data is incomplete.")
    print("- Guardrails prevent unsafe automation and force human review when needed.\n")


if __name__ == "__main__":
    main()
