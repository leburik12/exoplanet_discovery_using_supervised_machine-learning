import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE


class ModelEvaluator:
    def __init__(self, models: dict, X: pd.DataFrame, y: pd.Series):
        """
        Initializes the ModelEvaluator.

        Args:
          models (dict): A dictionary where keys are model names (str)
                         and values are model instances (e.g., KNNTrainer)
          X (pd.DataFrame): The feature matrix for the entire dataset
          y (pd.Series): The target vector for the entire dataset.
        """
        self.models = (
            models  # {'knn': knn_trainer, 'svm': svm_trainer, 'xgb': xgb_trainer}
        )
        self.X = X
        self.y = y
        self.results = {}  # Stores aggregated metrics across folds for each model
        self.fold_results = {}  # Stores detailed results for each fold

    def evaluate_all(self, cv, verbose: bool = True):
        """
        Performs stratified k-fold cross-validation for all registered models.

        Args:
           cv (StratifiedKFold): The StratifiedKFold cross-validation splitter.
           verbose(bool): If True, prints detailed progress and fold metrics.
        """

        if not isinstance(cv, StratifiedKFold):
            raise TypeError("The 'cv' must be instance of StratifiedKFold")

        for name, model in self.models.items():
            if verbose:
                print(f"\n--- Initiating Cross-Validation for: {name.upper()} ---")

            fold_metrics_list = []
            roc_curves = []
            precision_recall_curves = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y), 1):
                if verbose:
                    print(f"\n Fold {fold}/{cv.n_splits}")

                # Split data for the current fold
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

                # Apply SMOTE only on the training data.
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(
                    X_train, y_train
                )

                # Fit the model on the training data of the current fold
                model.fit(X_train_resampled, y_train_resampled)

                # Predict probabilities and labels on the validation data
                y_proba = model.predict_proba(X_val)

                # For binary classification, ensure y_proba is correctly sliced for metrics
                if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                    y_pred = np.argmax(y_proba, axis=1)
                    y_proba_positive_class = y_proba[
                        :, 1
                    ]  # Probability of the positive class
                else:
                    y_pred = (y_proba > 0.5).astype(int).flatten()
                    y_proba_positive_class = y_proba.flatten()

                # Evaluate metrics for the current fold
                fold_metrics, roc_data, pr_data = self._evaluate_metrics(
                    y_val, y_pred, y_proba, y_proba_positive_class, name
                )
                fold_metrics_list.append(fold_metrics)
                roc_curves.append(roc_data)
                precision_recall_curves.append(pr_data)

                if verbose:
                    print("  Fold Metrics:")
                    for k, v in fold_metrics.items():
                        if isinstance(v, (int, float)):
                            print(f"   {k}: {v:.4f}")
                        elif k not in [
                            "confusion_matrix",
                            "classification_report",
                            "roc_curve_data",
                            "precision_recall_curve_data",
                        ]:
                            print(f"    {k}: {v}")  # Print other non-scalar metrics

            # Aggregate results for the current model across all folds
            avg_metrics = self._aggregate_fold_metrics(fold_metrics_list)
            self.results[name] = avg_metrics
            self.fold_results[name] = {
                "individual_folds": fold_metrics_list,
                "roc_curves": roc_curves,
                "precision_recall_curves": precision_recall_curves,
            }

            if verbose:
                print(f"\n--- Mean Cross-Validated Metrics for {name.upper()}: ---")
                for k, v in avg_metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}, {v:.4f}")
                    else:
                        print(f"   {k}: {v}")

    def _evaluate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        y_proba_positive_class: np.ndarray,
        model_name: str,
    ):
        """
        Args:
          y_true (pd.Series): The true labels for the validation set.
          y_pred (np.ndarray): The predicted labels for the validation set.
          y_proba (np.ndarray): The predicted probabilities for all classes.
          y_proba_positive_class (np.ndarray): The predicted probabilities for the positive class.
          model_name (str): The name of the model beign evaluated.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
            "log_loss": log_loss(y_true, y_proba),
            "brier_score_loss": brier_score_loss(y_true, y_proba_positive_class),
            "roc_auc": roc_auc_score(y_true, y_proba_positive_class),
            "average_precision": average_precision_score(
                y_true, y_proba_positive_class
            ),
        }

        # Store data for ROC curve plotting
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba_positive_class)
        roc_curve_data = {
            "fpr": fpr.tolist(),  # False Positive Rates
            "tpr": tpr.tolist(),  # True Positive Rate (Recall)
            "thresholds": thresholds_roc.tolist(),
            "y_true": y_true.tolist(),
            "y_score": y_proba_positive_class.tolist(),
            "model_name": model_name,
        }

        # Store data for Precision-Recall curve plotting
        precision, recall, thresholds_pr = precision_recall_curve(
            y_true, y_proba_positive_class
        )

        precision_recall_curve_data = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds_pr.tolist(),
            "model_name": model_name,
        }

        return metrics, roc_curve_data, precision_recall_curve_data

    def _aggregate_fold_metrics(self, fold_metrics_list: list):
        """
        Aggregates metrics from all folds by computing their mean.
        Handles both scalar and dictionary metrics (like classification_report).

        Args:
            fold_metrics_list (list): A list of dictionaries, where each dictionary
                                      contains metrics for a single fold.
        Returns:
           dict: A dictionary of aggregated (mean) metrics.
        """
        aggregated = {}
        # Initialize with the first fold's keys
        for key in fold_metrics_list[0].keys():
            if key not in [
                "confusion_matrix",
                "classification_report",
                "roc_curve_data",
                "precision_recall_curve_data",
            ]:
                if isinstance(fold_metrics_list[0][key], (int, float)):
                    aggregated[key] = np.mean(
                        [
                            fm[key]
                            for fm in fold_metrics_list
                            if isinstance(fm[key], (int, float))
                        ]
                    )
                else:
                    # For non-scalar values (e.g., 'N/A' for multiclass), just take the first value
                    aggregated[key] = fold_metrics_list[0][key]
            elif key == "classification_report":
                class_report_sum = {}
                count = 0
                for fm in fold_metrics_list:
                    if isinstance(
                        fm.get(key), dict
                    ):  # Ensure it's a dict before iterating
                        count += 1
                        for class_label, metrics_dict in fm[key].items():
                            if not isinstance(metrics_dict, dict):
                                continue  # Skip scalar entries like 'accuracy'

                            if class_label not in class_report_sum:
                                class_report_sum[class_label] = {
                                    m_key: [] for m_key in metrics_dict.keys()
                                }
                            for m_key, value in metrics_dict.items():
                                class_report_sum[class_label][m_key].append(value)

                averaged_class_report = {}
                for class_label, metrics_list_dict in class_report_sum.items():
                    averaged_class_report[class_label] = {
                        m_key: np.mean(values) if values else None
                        for m_key, values in metrics_list_dict.items()
                    }
                aggregated[key] = averaged_class_report
        return aggregated

    def summarize_result(
        self, sort_by: str = "roc_auc", ascending: bool = False
    ) -> pd.DataFrame:
        """
        Summarized the cross-validation results for all models.

         Args:
            sort_by (str): The metric by which to sort the results.
            ascending (bool): Whether to sort in ascending order.

         Returns:
            pd.DataFrame: A DataFrame summarizing the performance of each model.
        """
        if not self.results:
            print("No evaluation results available. Please run evaluate_all() first.")
            return pd.DataFrame()

        summary_df = pd.DataFrame(self.results).T
        if sort_by in summary_df.columns and pd.api.types.is_numeric_dtype(
            summary_df[sort_by]
        ):
            return summary_df.sort_values(sort_by, ascending=ascending)
        else:
            print(f"Warning: Cannot sort by '{sort_by}' | it's not numeric column.")
            return summary_df

    def plot_roc_curves(self):
        """
        Plots ROC curves for all models across all folds.
        """
        if not self.fold_results:
            print(
                "No fold results avaialble to plot ROC curves.Run evaluate_all() first."
            )
            return

        plt.figure(figsize=(12, 10))

        for model_name, data in self.fold_results.items():
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            # Plot individual fold ROC curves
            for i, roc_data in enumerate(data["roc_curves"]):
                if roc_data:
                    fpr = np.array(roc_data["fpr"])
                    tpr = np.array(roc_data["tpr"])
                    y_true = np.array(roc_data["y_true"])
                    y_score = np.array(roc_data["y_score"])

                    auc = roc_auc_score(y_true, y_score)
                    aucs.append(auc)

                    interp_tpr = np.interp(mean_fpr, fpr, tpr)

                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)

                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = np.mean(aucs)

                    plt.plot(
                        mean_fpr,
                        mean_tpr,
                        color="blue",
                        linestyle="--",
                        linewidth=2,
                        label=f"Mean {model_name} ROC (AUC = {mean_auc:.2f})",
                    )

            # Plot mean ROC curve
            if tprs:
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0  # The ROC curve is defined to end at (FPR=1, TPR=1)
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)

                plt.plot(
                    mean_fpr,
                    mean_tpr,
                    label=f"Mean {model_name} ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
                    linewidth=2,
                    linestyle="--",
                )

        # Reference line
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

        plt.xlabel("False Positive Rate (FPR)", fontsize=13)
        plt.ylabel("True Positive Rate (TPR)", fontsize=13)
        plt.title("Cross-Validated ROC Curves by Model", fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curves(self):
        """
        Plots Precision-Recall curves for all models across all folds.
        """
        if not self.fold_results:
            print(
                "No fold results available to plot Precision-Recall curves. Run evaluate_all() first."
            )
            return

        plt.figure(figsize=(10, 8))

        for model_name, data in self.fold_results.items():
            for i, pr_data in enumerate(data["precision_recall_curves"]):
                if pr_data:
                    plt.plot(
                        pr_data["recall"],
                        pr_data["precision"],
                        alpha=0.3,
                        label=f"{model_name} Fold {i+1}",
                    )

            # Plot the mean Precision-Recall curve for each model if available
            mean_recall = np.linspace(0, 1, 100)
            precisions = []
            for pr_data in data["precision_recall_curves"]:
                if pr_data:
                    # Interpolate precision at standard recall points
                    # np.interp expects the x-values to be monotonically increasing.
                    interp_precision = np.interp(
                        mean_recall, pr_data["recall"][::-1], pr_data["precision"][::-1]
                    )
                    precisions.append(interp_precision)
            if precisions:
                mean_precision = np.mean(precisions, axis=0)
                plt.plot(
                    mean_recall,
                    mean_precision,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean {model_name} PR Curve",
                )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves Across Folds")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
