import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class EvaluateMetrics:
    def __init__(self):
        self.targets = None
        self.predicts = None

    def _get_accuracy(self) -> dict:
        return {"accuracy": accuracy_score(self.targets, self.predicts)}
    

    def _get_f1_score(self) -> dict:
        return {"f1_score": f1_score(self.targets, self.predicts)}
    

    def _get_precision(self) -> dict:
        return {"precision": precision_score(self.targets, self.predicts)}
    

    def _get_recall(self) -> dict:
        return {"recall": recall_score(self.targets, self.predicts)}


    def show_summary(self, metric_name: str, metrics_dict: dict):
        print("---" * 15)
        print(f"\t\t{metric_name.upper()}")
        print(f"- Accuracy :: ", metrics_dict["accuracy"])
        print(f"- F1 Score :: ", metrics_dict["f1_score"])
        print(f"- Precision :: ", metrics_dict["precision"])
        print(f"- Recall :: ", metrics_dict["recall"])
        print("---" * 15, "\n")


    def get_scores(
        self,
        predicts: np.ndarray,
        targets: np.ndarray,
    ):
        assert (predicts.ndim == 1) and (targets.ndim == 1), "Some array is not 1D"

        self.targets = targets
        self.predicts = predicts

        metrics = {}
        metrics.update(self._get_accuracy())
        metrics.update(self._get_f1_score())
        metrics.update(self._get_precision())
        metrics.update(self._get_recall())

        return metrics