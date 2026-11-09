"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import os
import sys
import glob
for p in glob.glob('/content/**/grader_contracts', recursive=True):
    parent_dir = os.path.dirname(p)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test."""
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)
    y_test = np.array(data.test_target, dtype=float)

    seed = getattr(data, 'random_seed', 42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=seed,
        n_jobs=-1
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=seed
    )

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('hgb', hgb)],
        voting='soft',
        n_jobs=-1
    )

    ensemble.fit(X_train_scaled, y_train)

    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    y_pred = ensemble.predict(X_test_scaled)

    roc_auc = float(roc_auc_score(y_test, y_proba))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    return EnsembleResult(
        model_name="Voting (RandomForest + HistGradientBoosting)",
        probabilities=y_proba.tolist(),
        predictions=y_pred.tolist(),
        roc_auc=roc_auc,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1
    )
