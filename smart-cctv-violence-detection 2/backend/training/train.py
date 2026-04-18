# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import os
# import sys
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# sys.path.append('..')
# from config import *
# from temporal_model.lstm_model import ViolenceDetectionModel
# from preprocessing.normalization import create_dataloaders

# class Trainer:
#     def __init__(self, model, train_loader, test_loader, device=DEVICE):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.device = device
        
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
#         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
        
#         self.train_losses = []
#         self.train_accuracies = []
#         self.test_losses = []
#         self.test_accuracies = []
    
#     def train_epoch(self):
#         self.model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
#             data, target = data.to(self.device), target.to(self.device)
            
#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
#             pred = output.argmax(dim=1)
#             correct += pred.eq(target).sum().item()
#             total += target.size(0)
        
#         avg_loss = total_loss / len(self.train_loader)
#         accuracy = 100. * correct / total
        
#         return avg_loss, accuracy
    
#     def test_epoch(self):
#         self.model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for data, target in tqdm(self.test_loader, desc="Testing"):
#                 data, target = data.to(self.device), target.to(self.device)
#                 output = self.model(data)
#                 loss = self.criterion(output, target)
                
#                 total_loss += loss.item()
#                 pred = output.argmax(dim=1)
#                 correct += pred.eq(target).sum().item()
#                 total += target.size(0)
        
#         avg_loss = total_loss / len(self.test_loader)
#         accuracy = 100. * correct / total
        
#         return avg_loss, accuracy
    
#     def train(self, num_epochs=NUM_EPOCHS):
#         best_accuracy = 0
        
#         for epoch in range(num_epochs):
#             print(f"\nEpoch {epoch+1}/{num_epochs}")
            
#             train_loss, train_acc = self.train_epoch()
#             test_loss, test_acc = self.test_epoch()
            
#             self.scheduler.step()
            
#             self.train_losses.append(train_loss)
#             self.train_accuracies.append(train_acc)
#             self.test_losses.append(test_loss)
#             self.test_accuracies.append(test_acc)
            
#             print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#             print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
#             # Save best model
#             if test_acc > best_accuracy:
#                 best_accuracy = test_acc
#                 self.save_model(f"best_model_acc_{test_acc:.2f}.pth")
            
#             # Save checkpoint every 10 epochs
#             if (epoch + 1) % 10 == 0:
#                 self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
        
#         print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")
#         self.plot_training_history()
    
#     def save_model(self, filename):
#         os.makedirs("models", exist_ok=True)
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'train_losses': self.train_losses,
#             'train_accuracies': self.train_accuracies,
#             'test_losses': self.test_losses,
#             'test_accuracies': self.test_accuracies
#         }, os.path.join("models", filename))
#         print(f"Model saved: {filename}")
    
#     def plot_training_history(self):
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
#         ax1.plot(self.train_losses, label='Train Loss')
#         ax1.plot(self.test_losses, label='Test Loss')
#         ax1.set_title('Training and Test Loss')
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Loss')
#         ax1.legend()
        
#         ax2.plot(self.train_accuracies, label='Train Accuracy')
#         ax2.plot(self.test_accuracies, label='Test Accuracy')
#         ax2.set_title('Training and Test Accuracy')
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Accuracy (%)')
#         ax2.legend()
        
#         plt.tight_layout()
#         plt.savefig('training_history.png')
#         plt.show()

# def main():
#     print("Loading data...")
#     train_loader, test_loader = create_dataloaders()
    
#     print("Initializing model...")
#     model = ViolenceDetectionModel()
    
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
#     print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
#     print("Starting training...")
#     trainer = Trainer(model, train_loader, test_loader)
#     trainer.train()

# if __name__ == "__main__":
#     main()


import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Fix import paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config import *
from temporal_model.lstm_model import ViolenceDetectionModel
from preprocessing.normalization import create_dataloaders


class Trainer:
    def __init__(self, model, train_loader, test_loader, device=DEVICE):
        self.device = device
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        # ✅ Correct loss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in tqdm(self.train_loader, desc="Training", leave=False):
            data = data.to(self.device)                   # [B, T, 3, H, W]
            target = target.to(self.device).unsqueeze(1) # [B, 1]

            self.optimizer.zero_grad()

            output = self.model(data)                     # [B, 1]
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def test_epoch(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing", leave=False):
                data = data.to(self.device)
                target = target.to(self.device).unsqueeze(1)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()

                preds = (torch.sigmoid(output) > 0.5).float()
                correct += (preds == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def save_model(self, filename):
        os.makedirs("models", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "train_accuracies": self.train_accuracies,
                "test_losses": self.test_losses,
                "test_accuracies": self.test_accuracies,
            },
            os.path.join("models", filename),
        )
        print(f"✅ Model saved: {filename}")

    def plot_history(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label="Train Accuracy")
        plt.plot(self.test_accuracies, label="Test Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.close()

    def train(self, num_epochs=EPOCHS):
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test_epoch()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model("best_model.pth")

        print(f"\n🏆 Best Test Accuracy: {best_acc:.2f}%")
        self.plot_history()


def main():
    print("🔹 Loading dataloaders...")
    train_loader, test_loader = create_dataloaders()

    print("🔹 Initializing model...")
    model = ViolenceDetectionModel()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = Trainer(model, train_loader, test_loader)
    trainer.train()


if __name__ == "__main__":
    main()
