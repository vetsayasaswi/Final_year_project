import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Fix import path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config import *
from temporal_model.lstm_model import ViolenceDetectionModel
from preprocessing.normalization import create_dataloaders


class ModelEvaluator:
    def __init__(self, model_path, device=DEVICE):
        self.device = device
        self.model = ViolenceDetectionModel().to(device)
        self.load_model(model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")

    def evaluate(self, test_loader):
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                logits = self.model(data)              # [B, 1]
                probs = torch.sigmoid(logits)          # [B, 1]
                preds = (probs > 0.5).int()             # [B, 1]

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return (
            np.array(all_preds).flatten(),
            np.array(all_labels).flatten(),
            np.array(all_probs).flatten()
        )

    def calculate_metrics(self, preds, labels, probs):
        report = classification_report(
            labels,
            preds,
            target_names=["NonFight", "Fight"],
            digits=4
        )

        cm = confusion_matrix(labels, preds)
        auc = roc_auc_score(labels, probs)

        return report, cm, auc

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["NonFight", "Fight"],
            yticklabels=["NonFight", "Fight"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

    def plot_roc_curve(self, labels, probs):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()


def main():
    print("🔹 Loading test data...")
    _, test_loader = create_dataloaders()

    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("❌ best_model.pth not found")
        return

    evaluator = ModelEvaluator(model_path)

    print("🔹 Evaluating model...")
    preds, labels, probs = evaluator.evaluate(test_loader)

    report, cm, auc = evaluator.calculate_metrics(preds, labels, probs)

    print("\n📊 CLASSIFICATION REPORT\n")
    print(report)
    print(f"\nROC AUC Score: {auc:.4f}")

    evaluator.plot_confusion_matrix(cm)
    evaluator.plot_roc_curve(labels, probs)

    print("\n✅ Evaluation complete")
    print("Saved: confusion_matrix.png, roc_curve.png")


if __name__ == "__main__":
    main()
