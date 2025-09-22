import os
import sqlite3
import pandas as pd
import numpy as np
import warnings
import concurrent.futures
from metrics import feature_and_label_extractor

class MetricDatabase:
    def __init__(self, db_path="metrics.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS metric_results")
        cur.execute("DROP TABLE IF EXISTS metric_aggregated")
        cur.execute("DROP TABLE IF EXISTS metric_recommendations")
        # Raw results table (with dataset column)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metric_results (
            metric_name TEXT,
            dataset TEXT,  -- 'train' or 'test'
            code TEXT,
            species TEXT,
            value BLOB
        )
        """)
        
        # Aggregates table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metric_aggregated (
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
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metric_recommendations (
            metric_name TEXT,
            fit_for_training BOOLEAN,
            n_samples INTEGER,
            n_classes INTEGER,
            class_distribution TEXT,
            reason TEXT
        )
        """)
        
        self.conn.commit()


    # -----------------------------
    # Add metric results
    # -----------------------------

    def add_metric_results(self, Metric, config_file, train_spectra, test_spectra, collection_list, plot=None):
        """
        Process a Metric with separate train and test spectra.
        - Train results are saved and aggregated.
        - Test results are saved but not aggregated.
        - Normalizations are computed in two flavors:
          ▸ train_normalized: dict-of-dicts (like test)
          ▸ train_normalized_real: against true species
        - A recommendation is generated from the train set.
        """
        metric_name = Metric.get_name()
        cur = self.conn.cursor()

        # ----------------------
        # TRAIN SPECTRA
        # ----------------------
        train_codes, train_features, train_labels = feature_and_label_extractor(
            Metric, config_file, train_spectra, collection_list, debug=False, plot=plot
        )
        train_results = list(zip(train_codes, train_features, train_labels))

        cur.executemany(
            """
            INSERT INTO metric_results (metric_name, dataset, code, species, value)
            VALUES (?, 'train', ?, ?, ?)
            """,
            [
                (metric_name, code, species,
                 float(val) if np.isscalar(val) else str(val))
                for code, val, species in train_results
            ]
        )
        self.conn.commit()

        # Aggregates only for train
        agg = None
        if train_results:
            df_train = pd.DataFrame(train_results, columns=["code", "metric", "species"])
            if np.isscalar(df_train["metric"].iloc[0]):
                agg = df_train.groupby("species")["metric"].agg(
                    ["mean", "std", "median", "min", "max", "count"]
                )
                agg["iqr"] = df_train.groupby("species")["metric"].apply(
                    lambda x: np.percentile(x, 75) - np.percentile(x, 25)
                )
                agg["cv"] = agg["std"] / agg["mean"]

                cur.executemany(
                    """
                    INSERT INTO metric_aggregated
                    (metric_name, species, mean, std, median, min, max, count, iqr, cv)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            metric_name,
                            species,
                            row["mean"], row["std"], row["median"],
                            row["min"], row["max"], int(row["count"]),
                            row["iqr"], row["cv"]
                        )
                        for species, row in agg.iterrows()
                    ]
                )
                self.conn.commit()

        # Recommendation (only on train)
        rec = self._generate_recommendation(train_results, metric_name)
        cur.execute(
            """
            INSERT INTO metric_recommendations
            (metric_name, fit_for_training, n_samples, n_classes, class_distribution, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                metric_name,
                rec["fit_for_training"],
                rec["n_samples"],
                rec["n_classes"],
                str(rec["class_distribution"]),
                str(rec["reason"]),
            )
        )
        self.conn.commit()

        # ----------------------
        # TEST SPECTRA
        # ----------------------
        test_codes, test_features, test_labels = feature_and_label_extractor(
            Metric, config_file, test_spectra, collection_list, debug=False, plot=plot
        )
        test_results = list(zip(test_codes, test_features, test_labels))

        cur.executemany(
            """
            INSERT INTO metric_results (metric_name, dataset, code, species, value)
            VALUES (?, 'test', ?, ?, ?)
            """,
            [
                (metric_name, code, species,
                 float(val) if np.isscalar(val) else str(val))
                for code, val, species in test_results
            ]
        )
        self.conn.commit()
        
        # ----------------------
        # NORMALIZATION SECTION
        # ----------------------
        if train_results and agg is not None:
            agg_dict = {species: row.to_dict() for species, row in agg.iterrows()}

            # two flavors for train
            train_normalized = self.normalize_train_results(train_results, agg_dict)
            train_normalized_real = self.normalize_train_results_real(train_results, agg_dict)

            # test: dict of dicts, always
            test_normalized = self._normalize_test_results(test_results, agg_dict)
        else:
            train_normalized, train_normalized_real, test_normalized = [], [], []

        result_dict = {
            "metric": metric_name,
            "train": train_results,
            "test": test_results,
            "train_normalized": train_normalized,               # dict of dicts
            "train_normalized_real": train_normalized_real,     # true species
            "test_normalized": test_normalized,
            "aggregated": agg.to_dict() if (agg is not None and train_results and np.isscalar(df_train["metric"].iloc[0])) else {},
            "recommendation": rec
        }

        # --- Validation step ---
        required_keys = ["metric", "train", "test", "train_normalized",
                         "train_normalized_real", "test_normalized", "aggregated", "recommendation"]
        for key in required_keys:
            if key not in result_dict:
                warnings.warn(f"[MetricDatabase] Key '{key}' missing in result_dict", UserWarning)
            elif result_dict[key] is None:
                warnings.warn(f"[MetricDatabase] Key '{key}' is None", UserWarning)

        return result_dict






    # -----------------------------
    # Retrieve data
    # -----------------------------
        # -----------------------------
    # Normalize results (new method)
    # -----------------------------
        # -----------------------------
        # -----------------------------
    # Normalize TRAIN results (dictionary of dictionaries)
    # -----------------------------
    def normalize_train_results(self, results, aggregated):
        """
        Normalize train results against every species' aggregates.
        Similar structure to test normalization.

        Parameters
        ----------
        results : list of (code, value, species)
        aggregated : dict {species: {"mean":..., "std":...,...}}

        Returns
        -------
        list of (code, {species: normalized_value}, true_species)
        """
        normalized = []
        for code, value, true_species in results:
            norm_dict = {}
            for species, stats in aggregated.items():
                mean, std = stats.get("mean"), stats.get("std")

                # ---- Scalar case ----
                if np.isscalar(value):
                    norm_val = (value - mean) / std if (std not in [0, None] and not np.isnan(std)) else np.nan
                else:
                    try:
                        vec = np.array(eval(value)) if isinstance(value, str) else np.array(value)
                        mean_vec = np.array(mean if isinstance(mean, (list, np.ndarray)) else [mean] * len(vec))
                        std_vec = np.array(std if isinstance(std, (list, np.ndarray)) else [std] * len(vec))
                        with np.errstate(divide="ignore", invalid="ignore"):
                            norm_val = (vec - mean_vec) / std_vec
                            norm_val[np.isnan(norm_val)] = np.nan
                    except Exception:
                        warnings.warn(f"Failed to normalize vector metric for {code} against {species}", UserWarning)
                        norm_val = np.nan

                norm_dict[species] = norm_val

            normalized.append((code, norm_dict, true_species))

        return normalized


    # -----------------------------
    # Normalize TRAIN results (true species only)
    # -----------------------------
    def normalize_train_results_real(self, results, aggregated):
        """
        Normalize train results using the correct species aggregate only.

        Parameters
        ----------
        results : list of (code, value, species)
        aggregated : dict {species: {"mean":..., "std":...,...}}

        Returns
        -------
        list of (code, normalized_value, species)
        """
        normalized = []
        for code, value, species in results:
            if species not in aggregated:
                warnings.warn(f"No aggregate stats for species '{species}', skipping {code}", UserWarning)
                continue

            mean, std = aggregated[species]["mean"], aggregated[species]["std"]

            # ---- Scalar case ----
            if np.isscalar(value):
                norm_val = (value - mean) / std if (std not in [0, None] and not np.isnan(std)) else np.nan
            else:
                try:
                    vec = np.array(eval(value)) if isinstance(value, str) else np.array(value)
                    mean_vec = np.array(mean if isinstance(mean, (list, np.ndarray)) else [mean] * len(vec))
                    std_vec = np.array(std if isinstance(std, (list, np.ndarray)) else [std] * len(vec))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        norm_val = (vec - mean_vec) / std_vec
                        norm_val[np.isnan(norm_val)] = np.nan
                except Exception:
                    warnings.warn(f"Failed to normalize vector metric for {code} in train (true species)", UserWarning)
                    norm_val = np.nan

            normalized.append((code, norm_val, species))

        return normalized



    # -----------------------------
    # Normalize TEST results
    # -----------------------------
    def _normalize_test_results(self, results, aggregated):
        """
        Normalize test results against every species' aggregates.

        Parameters
        ----------
        results : list of (code, value, species)
        aggregated : dict {species: {"mean":..., "std":...,...}}

        Returns
        -------
        list of (code, {species: normalized_value}, true_species)
        """
        normalized = []
        for code, value, true_species in results:
            norm_dict = {}
            for species, stats in aggregated.items():
                mean, std = stats.get("mean"), stats.get("std")

                # ---- Scalar case ----
                if np.isscalar(value):
                    norm_val = (value - mean) / std if (std not in [0, None] and not np.isnan(std)) else np.nan

                # ---- Vectorial case ----
                else:
                    try:
                        vec = np.array(eval(value)) if isinstance(value, str) else np.array(value)
                        mean_vec = np.array(mean if isinstance(mean, (list, np.ndarray)) else [mean]*len(vec))
                        std_vec = np.array(std if isinstance(std, (list, np.ndarray)) else [std]*len(vec))

                        with np.errstate(divide="ignore", invalid="ignore"):
                            norm_val = (vec - mean_vec) / std_vec
                            norm_val[np.isnan(norm_val)] = np.nan
                    except Exception:
                        warnings.warn(f"Failed to normalize vector metric for {code} against species {species}", UserWarning)
                        norm_val = np.nan

                norm_dict[species] = norm_val

            normalized.append((code, norm_dict, true_species))

        return normalized



    def get_results(self, metric_name):
        return pd.read_sql_query(
            "SELECT * FROM metric_results WHERE metric_name=?", self.conn, params=(metric_name,)
        )

    def get_aggregates(self, metric_name):
        return pd.read_sql_query(
            "SELECT * FROM metric_aggregated WHERE metric_name=?", self.conn, params=(metric_name,)
        )
    
    def _generate_recommendation(self, results, metric_name, min_samples=20, min_classes=2):
        """Generate recommendation about whether dataset is suitable for NN training."""
        if not results:
            return {"fit_for_training": False, "reason": "No results available."}
        
        df = pd.DataFrame(results, columns=["code", "metric", "species"])
        
        # Convert metric column to numeric, coercing errors to NaN
        df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
        
        # Check for NaN or infinite values in metric column
        valid_mask = ~df["metric"].isna() & ~df["metric"].apply(lambda x: np.isinf(x) if isinstance(x, (int, float)) else False)
        n_invalid = len(df) - valid_mask.sum()
        n_samples = valid_mask.sum()
        n_classes = df["species"][valid_mask].nunique()
        class_counts = df["species"][valid_mask].value_counts(normalize=True)
        
        fit, reasons = True, []
        if n_invalid > 0:
            fit = False
            reasons.append(f"Found {n_invalid} invalid (NaN or infinite) metric values")
        if n_samples < min_samples:
            fit = False
            reasons.append(f"Too few valid samples ({n_samples} < {min_samples})")
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
            "n_invalid_metrics": n_invalid,
            "class_distribution": class_counts.to_dict(),
            "reason": reasons if reasons else "Dataset looks balanced."
        }
    # -----------------------------
    # Dataset summary (fit check)
    # -----------------------------
    def dataset_summary(self, metric_name, min_samples=20, min_classes=2):
        df = self.get_results(metric_name)
        if df.empty:
            return {"fit_for_training": False, "reason": "No results available."}

        n_samples = len(df)
        n_classes = df["species"].nunique()
        class_counts = df["species"].value_counts(normalize=True)

        fit, reasons = True, []
        if n_samples < min_samples:
            fit, reasons = False, reasons + [f"Too few samples ({n_samples} < {min_samples})"]
        if n_classes < min_classes:
            fit, reasons = False, reasons + [f"Too few classes ({n_classes} < {min_classes})"]
        if (class_counts < 0.05).any():
            fit, reasons = False, reasons + ["At least one class underrepresented (<5%)"]

        return {
            "fit_for_training": fit,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "class_distribution": class_counts.to_dict(),
            "reason": reasons if reasons else "Dataset looks balanced."
        }

    # -----------------------------
    # Batch add (sequential + parallel)
    # -----------------------------
    
    
    def add_multiple_metrics(
        self, metric_list, config_file, train_spectra, test_spectra, collection_list, plot=None
    ):
        summary_report = {}
        results_for_every_metric = {}

        for Metric in metric_list:
            metric_name = Metric.get_name()
            try:
                results = self.add_metric_results(
                    Metric,
                    config_file,
                    train_spectra=train_spectra,
                    test_spectra=test_spectra,
                    collection_list=collection_list,
                    plot=plot,
                )
                print("results", results)

                summary_entry = {
                    "train_samples": len(results.get("train", [])),
                    "test_samples": len(results.get("test", [])),
                    "recommendation": results.get("recommendation"),
                }
                summary_report[metric_name] = summary_entry

                # merge results + summary_entry into one dict
                results_for_every_metric[metric_name] = {"metric_name": metric_name, **results, **summary_entry}

            except Exception as e:
                summary_report[metric_name] = {
                    "fit_for_training": False,
                    "reason": str(e),
                }
                results_for_every_metric[metric_name] = {
                    "error": str(e)
                }

        return summary_report, results_for_every_metric

