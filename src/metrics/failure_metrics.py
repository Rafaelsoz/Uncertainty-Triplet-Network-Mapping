import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

class FailureMetrics:
    def __init__(self, fixed_tpr: float = 0.95):

        self.fixed_tpr = fixed_tpr
        self.success_predictions = None
        self.error_predictions = None
        self.scores = None


    def _get_roc_auc_score(self) -> dict:
        roc_auc_score_value = roc_auc_score(self.success_predictions, self.scores)
        return {"roc_auc_score": roc_auc_score_value}


    def _get_area_under_precision_recall_curve(self, success_samples: bool) -> dict:
        if success_samples:
          aupr = average_precision_score(self.success_predictions, self.scores)
          return {"aupr_success": aupr}

        aupr = average_precision_score(self.error_predictions, -self.scores)
        return {"aupr_error": aupr}


    def _get_fpr_fixed_tpr(self, just_fpr: bool = True) -> dict:
        fpr, tpr, thresholds = roc_curve(self.success_predictions, self.scores)
        
        differences = np.abs(tpr - self.fixed_tpr)
        lower_diff_idx = np.argmin(differences)

        nearest_th = thresholds[lower_diff_idx]
        nearest_tpr = tpr[lower_diff_idx]
        nearest_fpr = fpr[lower_diff_idx]

        if just_fpr:
            return {
                f"fpr_at_{self.fixed_tpr}tpr":nearest_fpr
            }
        
        return {
            f"fpr_at_{self.fixed_tpr}tpr":{
                "nearest_th": nearest_th,
                "nearest_tpr": nearest_tpr,
                "nearest_fpr":nearest_fpr
            }
        }


    def show_summary(self, metric_name: str, metrics_dict: dict):
        print("---" * 15)
        print(f"\t\t{metric_name.upper()}")
        print(f"- FPR-95%-TPR :: ", metrics_dict[f"fpr_at_{self.fixed_tpr}tpr"])
        print(f"- ROC-AUC :: ", metrics_dict["roc_auc_score"])
        print(f"- AUPR-Error :: ", metrics_dict["aupr_error"])
        print(f"- AUPR-Success :: ", metrics_dict["aupr_success"])
        print("---" * 15, "\n")


    def get_scores(
        self,
        predicts: np.ndarray,
        targets: np.ndarray,
        scores: np.ndarray,
        just_fpr: bool = True
    ):
        assert (predicts.ndim == 1) and (targets.ndim == 1) and (scores.ndim == 1), "Some array is not 1D"

        self.success_predictions = predicts == targets
        self.error_predictions = ~self.success_predictions
        self.scores = scores

        metrics = {}
        metrics.update(self._get_roc_auc_score())
        metrics.update(self._get_area_under_precision_recall_curve(success_samples=True))
        metrics.update(self._get_area_under_precision_recall_curve(success_samples=False))
        metrics.update(self._get_fpr_fixed_tpr(just_fpr=just_fpr))

        return metrics
