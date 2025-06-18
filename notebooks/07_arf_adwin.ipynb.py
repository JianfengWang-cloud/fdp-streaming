#!/usr/bin/env python
# coding: utf-8
#
# Adaptive Random Forest (ARF) Champion Model Bake-Off v4 (Final)
#
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import optuna
from river import compose, ensemble, metrics, tree, drift

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ARF_Champion_Finder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.df = self._load_data(config['csv_path'])
        self.feat_cols = [c for c in self.df.columns if c not in config['meta_cols']]

        n = len(self.df)
        self.tune_df = self.df.iloc[:int(n * 0.8)]

    def _load_data(self, path: str | Path) -> pd.DataFrame:
        print("─" * 60 + "\n1. Loading and cleaning data...")
        df = pd.read_csv(path).loc[:, ~pd.read_csv(path).columns.duplicated()]
        req = set(self.config['meta_cols'])
        if missing := req - set(df.columns): raise KeyError(f"Missing cols: {missing}")
        df[self.config['quarter_col']] = pd.to_datetime(df[self.config['quarter_col']])
        df.sort_values([self.config['id_col'], self.config['quarter_col']], inplace=True)
        df = df.dropna()
        return df

    def _run_stream_evaluation(self, dataframe: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Helper function to run a prequential evaluation on a given dataframe."""
        from river import drift
        from river import tree, ensemble, metrics

        # 基模型 & 装袋森林
        base_model = tree.HoeffdingTreeClassifier(
            grace_period=params['grace_period'],
            delta=params['delta'],
            split_criterion='hellinger'
        )
        forest = ensemble.BaggingClassifier(
            model=base_model,
            n_models=params['n_models'],
            seed=self.config['seed']
        )

        # FIX: 直接实例化 ADWIN，update() 返回是否检测到漂移
        detector = drift.ADWIN()

        metric_suite = {
            "F1": metrics.F1(),
            "Precision": metrics.Precision(),
            "Recall": metrics.Recall(),
            "G-Mean": metrics.GeometricMean(),
            "AUC": metrics.ROCAUC()
        }

        for _, row in dataframe.iterrows():
            x = row[self.feat_cols].to_dict()
            y = int(row[self.config['target_col']])

            # 1) 预测
            y_prob_one = forest.predict_proba_one(x)
            y_pred = forest.predict_one(x)

            # 2) 更新指标
            if y_pred is not None:
                for name, metric in metric_suite.items():
                    if name == "AUC":
                        metric.update(y_true=y, y_pred=y_prob_one)
                    else:
                        metric.update(y_true=y, y_pred=y_pred)

                # 3) 误差 & 漂移检测
                error = int(y_pred != y)
                drift_detected = detector.update(error)
                if drift_detected:
                    # 森林 & 检测器都要重置
                    forest.reset()
                    detector = drift.ADWIN()

            # 4) 学习新样本
            forest.learn_one(x, y)

        # 收集并返回所有指标
        return {name: m.get() for name, m in metric_suite.items()}

    def _objective(self, trial: optuna.Trial) -> float:
        """The objective function for Optuna to maximize."""
        params = {
            'n_models': trial.suggest_int('n_models', 5, 25),
            'grace_period': trial.suggest_int('grace_period', 50, 400),
            'delta': trial.suggest_float('delta', 1e-7, 1e-2, log=True),
        }

        results = self._run_stream_evaluation(self.tune_df, params)
        return results.get("F1", 0.0)

    def _evaluate_champion_model(self, params: Dict[str, Any]):
        """Evaluates the champion ARF on the entire dataset."""
        print("\n--- Evaluating Champion ARF Model on the Full Data Stream (100%) ---")

        final_results = self._run_stream_evaluation(self.df, params)

        print(f"\n[Optuna-Tuned ARF+ADWIN] Final Cumulative Performance:")
        for name, value in final_results.items():
            print(f"  {name:<10} = {value:.4f}")

    def run(self):
        """Orchestrates the entire ARF bake-off process."""
        print("\n" + "═" * 60)
        print("Starting Adaptive Random Forest (ARF) Championship Bake-Off")
        print("═" * 60)

        print("4. Starting Optuna optimization process on the first 80% of data...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.config['optuna_trials'], show_progress_bar=True)

        print(f"\nOptuna process finished!")
        print(f"🏆 Best cumulative F1-score on Tune Stream: {study.best_value:.4f}")
        print(f"🏆 Best Hyperparameters Found: {study.best_params}")

        self._evaluate_champion_model(study.best_params)
        print("\nARF Bake-Off Complete!")


if __name__ == "__main__":
    CONFIG = {
        "csv_path": r"C:\Users\23661\Desktop\cvm_indicators_dataset_2011-2021.csv",
        "id_col": "ID", "quarter_col": "QUARTER", "target_col": "LABEL",
        "meta_cols": ["ID", "QUARTER", "LABEL"],
        "seed": 42,

        "optuna_trials": 50,
    }

    champion_finder = ARF_Champion_Finder(config=CONFIG)
    champion_finder.run()