import os

# Fix for OpenMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

# Import your model
from model import AlexNetCBAM


class PickleDataset(Dataset):
    """Custom Dataset for loading pickle datadicts"""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class PlantClassifierTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directories
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        # Class names
        self.class_names = []
        self.key_dict = None

        print(f"Using device: {self.device}")

    def get_transforms(self):
        """Define data augmentation and normalization transforms"""
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return train_transform, val_transform

    def load_data(self):
        """Load training and validation datasets from pickle files"""
        dataset_focus = self.config.get("dataset_focus", "all")

        # Load key_dict for label mapping
        print("Loading key dictionary...")
        with open("key_dict.pkl", "rb") as f:
            self.key_dict = pickle.load(f)

        # Determine which dataset to load based on focus
        if dataset_focus == "all":
            print(f"Loading dataset for all plants classification...")
            dataset_path = "all_plants_balanced_dataset.pkl"
            # For 'all' mode, we classify by plant type
            # Create reverse mapping from label index to plant name
            idx_to_plant = {v: k for k, v in self.key_dict["all"].items()}
            self.class_names = [idx_to_plant[i] for i in range(len(idx_to_plant))]
        else:
            # Load individual plant dataset for disease classification
            plant_name = dataset_focus.lower()
            print(f"Loading dataset for {plant_name} disease classification...")
            dataset_path = f"{plant_name}_balanced_dataset.pkl"
            # For individual plant mode, we classify by disease
            # Create reverse mapping from label index to disease name
            idx_to_disease = {v: k for k, v in self.key_dict[plant_name].items()}
            self.class_names = [idx_to_disease[i] for i in range(len(idx_to_disease))]

        # Load the pickle datadict
        print(f"Loading data from {dataset_path}...")
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        images = data["image"]
        labels = data["label"]

        print(f"Total samples: {len(images)}")
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {len(self.class_names)}")
        print(
            f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )

        # Split dataset into train and validation (80/20 split)
        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Get transforms
        train_transform, val_transform = self.get_transforms()

        # Create datasets
        train_dataset = PickleDataset(
            train_images, train_labels, transform=train_transform
        )

        val_dataset = PickleDataset(val_images, val_labels, transform=val_transform)

        # Determine num_workers based on platform
        # Windows has issues with multiprocessing and PIL images
        import platform

        num_workers = (
            0 if platform.system() == "Windows" else self.config["num_workers"]
        )

        if num_workers == 0:
            print(
                "Using num_workers=0 (single-process data loading for Windows compatibility)"
            )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader, self.class_names

    def build_model(self):
        """Initialize the AlexNet-CBAM model"""
        num_classes = len(self.class_names)
        model = AlexNetCBAM(num_classes=num_classes)
        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Compile model for faster training (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            print("Compiling model with torch.compile()...")
            # model = torch.compile(model)

        return model

    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc, all_preds, all_targets

    def plot_confusion_matrix(self, y_true, y_pred, classes, epoch):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        save_path = self.output_dir / f"confusion_matrix_epoch_{epoch+1}.png"
        plt.savefig(save_path)
        plt.close()

        return cm

    def save_checkpoint(self, model, optimizer, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": self.config,
            "class_names": self.class_names,
            "key_dict": self.key_dict,
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved with validation accuracy: {val_acc:.2f}%")

    def train(self):
        """Main training loop"""
        # Load data
        train_loader, val_loader, classes = self.load_data()

        # Build model
        model = self.build_model()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        # Training history
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_acc = 0.0

        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50 + "\n")

        for epoch in range(self.config["epochs"]):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate(
                model, val_loader, criterion
            )

            # Update learning rate
            scheduler.step(val_acc)

            # Save history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Log to tensorboard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar(
                "Learning_Rate", optimizer.param_groups[0]["lr"], epoch
            )

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            self.save_checkpoint(model, optimizer, epoch, val_acc, is_best)

            # Plot confusion matrix every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_confusion_matrix(val_targets, val_preds, classes, epoch)

            # Generate classification report
            if (epoch + 1) % 5 == 0:
                report = classification_report(
                    val_targets, val_preds, target_names=classes, digits=4
                )
                print("\nClassification Report:")
                print(report)

        # Final confusion matrix
        self.plot_confusion_matrix(
            val_targets, val_preds, classes, self.config["epochs"] - 1
        )

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        # Plot training curves
        self.plot_training_curves(history)

        print("\n" + "=" * 50)
        print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
        print("=" * 50 + "\n")

        self.writer.close()

    def plot_training_curves(self, history):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(history["train_acc"], label="Train Acc")
        ax2.plot(history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path = self.output_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300)
        plt.close()


def main():
    config_all = {
        "dataset_focus": "all",  # Options: 'all', 'soy', 'corn', 'wheat', 'rice', 'potato'
        "output_dir": "outputs/all_plants",
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 4,
    }

    trainer = PlantClassifierTrainer(config_all)
    trainer.train()
    for plant in ["soy", "corn", "wheat", "rice", "potato"]:
        config_all = {
            "dataset_focus": plant,
            "output_dir": f"outputs/{plant}",
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 4,
        }

        trainer = PlantClassifierTrainer(config_all)
        trainer.train()


if __name__ == "__main__":
    main()
