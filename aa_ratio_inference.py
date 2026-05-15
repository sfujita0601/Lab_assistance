from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Mapping

import pandas as pd


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"


AA20_ORDER = [
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Cystine",
    "Phe",
    "Tyr",
    "Thr",
    "Trp",
    "Val",
    "His",
    "Arg",
    "Ala",
    "Asp",
    "Asn",
    "Glu",
    "Gly",
    "Pro",
    "Ser",
    "Gln",
]

# Alias normalization for model compatibility.
# Model files use "Cys2", while user input often uses "Cystine" or "Cys".
AA_ALIAS = {
    "Cystine": "Cys2",
    "Cys": "Cys2",
    "Cys2": "Cys2",
}


@dataclass
class AARatioPredictor:
    """Predict adjusted amino-acid composition ratio from delta ratios.

    Input:
    - self_vs_ideal_diff_ratio: dict-like with amino-acid names as keys.
      Example keys: Ile, Leu, ..., Gln, and Cystine (or Cys/Cys2).

    Output:
    - pandas.Series of predicted adjusted composition ratio for 20 AAs.
    """

    model_dir: Path

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.models = self._load_models(self.model_dir)

    @staticmethod
    def _load_models(model_dir: Path) -> Dict[str, object]:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_paths = sorted(model_dir.glob("model_*.pkl"))
        if not model_paths:
            raise FileNotFoundError(f"No model_*.pkl found in: {model_dir}")

        models: Dict[str, object] = {}
        for p in model_paths:
            aa_name = p.stem.replace("model_", "", 1)
            with p.open("rb") as f:
                models[aa_name] = pickle.load(f)
        return models

    @staticmethod
    def _normalize_input_keys(diff_ratio: Mapping[str, float]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for key, value in diff_ratio.items():
            canonical = AA_ALIAS.get(key, key)
            normalized[canonical] = float(value)
        return normalized

    def predict_adjusted_ratio(self, self_vs_ideal_diff_ratio: Mapping[str, float]) -> pd.Series:
        """Run all models and return adjusted AA composition ratio predictions.

        The function uses each model's feature_names_in_ to build the exact
        per-model input vector, so feature order and model-specific subsets are
        handled automatically.
        """
        x_dict = self._normalize_input_keys(self_vs_ideal_diff_ratio)

        preds: Dict[str, float] = {}
        for target_aa, model in self.models.items():
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is None:
                raise ValueError(
                    f"Model for {target_aa} does not have feature_names_in_. "
                    "Please retrain/save with scikit-learn version that stores feature names."
                )

            missing = [f for f in feature_names if f not in x_dict]
            if missing:
                raise ValueError(
                    f"Missing input features for {target_aa}: {missing}. "
                    "Check input AA names and aliases."
                )

            x_row = pd.DataFrame([{f: x_dict[f] for f in feature_names}], columns=feature_names)
            pred = float(model.predict(x_row)[0])
            preds[target_aa] = pred

        # Keep output order user-friendly (AA20_ORDER), then append any extras.
        ordered = [aa for aa in AA20_ORDER if aa in preds]
        extras = sorted([aa for aa in preds.keys() if aa not in ordered])
        out_index = ordered + extras

        return pd.Series({aa: preds[aa] for aa in out_index}, name="adjusted_ratio")


def predict_adjusted_aa_composition(
    self_vs_ideal_diff_ratio: Mapping[str, float],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.Series:
    """Simple one-shot API.

    Parameters
    ----------
    self_vs_ideal_diff_ratio:
        Dict-like AA delta ratio input.
    model_dir:
        Directory containing model_*.pkl files.

    Returns
    -------
    pd.Series
        Predicted adjusted amino-acid composition ratio.
    """
    predictor = AARatioPredictor(Path(model_dir))
    return predictor.predict_adjusted_ratio(self_vs_ideal_diff_ratio)


def concentrations_to_diff_ratio(
    self_concentration: Mapping[str, float],
    ideal_concentration: Mapping[str, float],
    mode: str = "percent",
    eps: float = 1e-12,
) -> pd.Series:
    """Convert self/ideal AA concentrations to model input diff ratio.

    Parameters
    ----------
    self_concentration:
        Dict-like concentration values for self.
    ideal_concentration:
        Dict-like concentration values for ideal target.
    mode:
        "percent": (self - ideal) / ideal * 100
        "ratio":   self / ideal
    eps:
        Small value to avoid division by zero.

    Returns
    -------
    pd.Series
        Diff-ratio values indexed by AA20_ORDER.
    """
    self_s = pd.Series(self_concentration, dtype=float)
    ideal_s = pd.Series(ideal_concentration, dtype=float)

    # Cys alias normalization for alignment with model feature names.
    self_s = self_s.rename(index=AA_ALIAS)
    ideal_s = ideal_s.rename(index=AA_ALIAS)

    # Ensure all 20 AAs are present in output order.
    out_index = [AA_ALIAS.get(aa, aa) for aa in AA20_ORDER]
    self_s = self_s.reindex(out_index)
    ideal_s = ideal_s.reindex(out_index)

    if self_s.isna().any() or ideal_s.isna().any():
        missing_self = self_s[self_s.isna()].index.tolist()
        missing_ideal = ideal_s[ideal_s.isna()].index.tolist()
        raise ValueError(
            f"Missing concentration values. self missing={missing_self}, ideal missing={missing_ideal}"
        )

    denom = ideal_s.replace(0.0, eps)
    if mode == "percent":
        diff = (self_s - ideal_s) / denom * 100.0
    elif mode == "ratio":
        diff = self_s / denom
    else:
        raise ValueError("mode must be 'percent' or 'ratio'")

    # Convert index back to user-friendly amino acid names where possible.
    reverse_alias = {v: k for k, v in AA_ALIAS.items() if k == "Cystine"}
    diff.index = [reverse_alias.get(i, i) for i in diff.index]
    diff.name = "diff_ratio"
    return diff


if __name__ == "__main__":
    # Minimal runnable example.
    # Replace values with your own "self vs ideal" delta ratios.
    sample_input = {
        "Ile": 5.0,
        "Leu": -3.0,
        "Lys": 1.2,
        "Met": 0.0,
        "Cystine": -2.1,
        "Phe": 4.5,
        "Tyr": -1.0,
        "Thr": 0.3,
        "Trp": 2.2,
        "Val": -0.8,
        "His": 1.1,
        "Arg": -2.4,
        "Ala": 0.7,
        "Asp": -1.9,
        "Asn": 0.4,
        "Glu": 3.3,
        "Gly": -0.2,
        "Pro": 1.8,
        "Ser": -0.6,
        "Gln": 2.0,
    }

    pred = predict_adjusted_aa_composition(sample_input)
    print(pred)
