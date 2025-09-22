import os
import sqlite3
import pandas as pd
import numpy as np
import warnings

from metrics import feature_and_label_extractor
from feature_engineering import get_aggregated_data


class MetricDatabase:
    def __init__(self, db_path="metrics.sqlite"):
        self.db = {}
        self.db_path = db_path
        self._init_sqlite()

    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Raw datapoints (train/test flag included)
        c.execute("""
        CREATE TABLE IF NOT EXISTS metric_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT,
            dataset TEXT,  -- 'train' or 'test'
            code TEXT,
            value BLOB,
            species TEXT
        )
        """)

        # Aggregated stats (only train set)
        c.execute("""
        CREATE TABLE IF NOT EXISTS metric_aggregated (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT,
            species TEXT,
            mean REAL,
            std REAL,
            median REAL,
            min REAL,
            max REAL,
            count INTEGER,
            iqr REAL,
            cv REAL
        )
        """)

        # Recommendations table
        c.execute("""
        CREATE TABLE IF NOT EXISTS metric_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT,
            fit_for_training BOOLEAN,
            n_samples INTEGER,
            n_classes INTEGER,
            class_distribution TEXT,
            reason TEXT
        )
        """)

        conn.commit()
        conn.close()

    def add_metric_results(self, Metric, config_file, train_spectra, test_spectra, collection_list, plot=None):
        """Extract results for train/test separately. Aggregate and recommend only using train."""
        metric_name = Metric.get_name()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # ---- TRAIN SPECTRA ----
        train_codes, train_features, train_labels = feature_and_label_extractor(
            Metric, config_file, train_spectra, collection_list, debug=False, plot=plot
        )
        train_results = list(zip(train_codes, train_features, train_labels))
        self.db.setdefault(metric_name, {})["train"] = train_results

        c.executemany(
            "INSERT INTO metric_results (metric_name, dataset, code, value, species) VALUES (?, ?, ?, ?, ?)",
            [(metric_name, "train", code, str(val), species) for code, val, species in train_results]
        )

        # Aggregated stats
        aggregated = get_aggregated_data(Metric, train_spectra, config_file, collection_list, plot=plot)
        self.db[metric_name]["aggregated"] = aggregated
        if not aggregated.empty:
            aggregated_reset = aggregated.reset_index()
            for _, row in aggregated_reset.iterrows():
                c.execute("""
                INSERT INTO metric_aggregated
                (metric_name, species, mean, std, median, min, max, count, iqr, cv)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_name, row["species"],
                    row.get("mean"), row.get("std"), row.get("median"),
                    row.get("min"), row.get("max"), row.get("count"),
                    row.get("iqr"), row.get("cv")
                ))

        # Recommendation
        rec = self._generate_recommendation(train_results, metric_name)
        c.execute("""
        INSERT INTO metric_recommendations
        (metric_name, fit_for_training, n_samples, n_classes, class_distribution, reason)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            metric_name,
            rec["fit_for_training"],
            rec["n_samples"],
            rec["n_classes"],
            str(rec["class_distribution"]),
            str(rec["reason"])
        ))

        # ---- TEST SPECTRA ----
        test_codes, test_features, test_labels = feature_and_label_extractor(
            Metric, config_file, test_spectra, collection_list, debug=False, plot=plot
        )
        test_results = list(zip(test_codes, test_features, test_labels))
        self.db[metric_name]["test"] = test_results

        c.executemany(
            "INSERT INTO metric_results (metric_name, dataset, code, value, species) VALUES (?, ?, ?, ?, ?)",
            [(metric_name, "test", code, str(val), species) for code, val, species in test_results]
        )

        conn.commit()
        conn.close()
        return {"train": train_results, "test": test_results, "recommendation": rec}

    def _generate_recommendation(self, results, metric_name, min_samples=30, min_classes=2):
        """Generate training suitability recommendation."""
        if not results:
            return {"fit_for_training": False, "reason": "No results available."}

        df = pd.DataFrame(results, columns=["code", "metric", "species"])

        n_samples = len(df)
        n_classes = df["species"].nunique()
        class_counts = df["species"].value_counts(normalize=True)

        fit = True
        reasons = []
        if n_samples < min_samples:
            fit = False
            reasons.append(f"Too few samples ({n_samples} < {min_samples})")
        if n_classes < min_classes:
            fit = False
            reasons.append(f"Too few classes ({n_classes} < {min_classes})")
        if (class_counts < 0.05).any():
            fit = False
            reasons.append("At least one class underrepresented (<5%)")

        return {
            "fit_for_training": fit,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "class_distribution": class_counts.to_dict(),
            "reason": reasons if reasons else "Dataset looks balanced."
        }

    def get_results(self, metric_name, dataset="train"):
        return self.db.get(metric_name, {}).get(dataset, [])

    def get_aggregated(self, metric_name):
        return self.db.get(metric_name, {}).get("aggregated", pd.DataFrame())

    def get_recommendation(self, metric_name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM metric_recommendations WHERE metric_name=?", (metric_name,))
        row = c.fetchone()
        conn.close()
        return row
