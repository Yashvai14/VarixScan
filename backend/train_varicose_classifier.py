"""
High-Performance Varicose Vein Binary Classifier
Target: 95%+ Overall Accuracy with 90%+ Varicose Recall

Features:
- Transfer learning with EfficientNet-B3
- Advanced data augmentation
- Class imbalance handling with weighted loss
- Learning rate scheduling with warm restarts
- Early stopping and model checkpointing
- Comprehensive evaluation metrics
- Optimized threshold selection for high recall
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import timm  # For EfficientNet

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import glob
import random
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class AdvancedVaricoseDataset(Dataset):
    """Advanced dataset with comprehensive augmentation for varicose vein detection"""
    
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
        # Advanced augmentation pipeline
        if is_training:
            self.albumentations_transform = A.Compose([
                # Spatial transformations
                A.Rotate(limit=30, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.8
                ),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
                
                # Color augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                
                # Medical-specific augmentations
                A.ChannelShuffle(p=0.2),
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32,
                    min_holes=1, min_height=8, min_width=8,
                    fill_value=0, p=0.3
                ),
                
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.albumentations_transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (384, 384))  # EfficientNet-B3 optimal size
            
            # Apply albumentations
            if self.albumentations_transform:
                augmented = self.albumentations_transform(image=image)
                image = augmented['image']
            
            # Apply PyTorch transforms if any
            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = torch.zeros(3, 384, 384)
            label = torch.tensor(0, dtype=torch.long)
            return image, label

class EfficientNetVaricoseClassifier(nn.Module):
    """EfficientNet-based classifier optimized for varicose vein detection"""
    
    def __init__(self, num_classes=2, model_name='efficientnet_b3', pretrained=True, dropout_rate=0.5):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            drop_rate=dropout_rate
        )
        
        # Get the number of features from backbone
        self.num_features = self.backbone.num_features
        
        # Custom classifier head with additional regularization
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class VaricoseTrainer:
    """Comprehensive trainer for varicose vein classification"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = EfficientNetVaricoseClassifier(
            num_classes=2,
            model_name='efficientnet_b3',
            dropout_rate=config.get('dropout_rate', 0.5)
        ).to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_varicose_recall = 0.0
        
    def prepare_data(self, data_dir):
        """Prepare training and validation datasets"""
        
        # Find all images
        varicose_images = glob.glob(os.path.join(data_dir, 'varicose', '*.jpg')) + \
                         glob.glob(os.path.join(data_dir, 'varicose', '*.png')) + \
                         glob.glob(os.path.join(data_dir, 'varicose', '*.jpeg'))
        
        normal_images = glob.glob(os.path.join(data_dir, 'normal', '*.jpg')) + \
                       glob.glob(os.path.join(data_dir, 'normal', '*.png')) + \
                       glob.glob(os.path.join(data_dir, 'normal', '*.jpeg'))
        
        # Create labels
        varicose_labels = [1] * len(varicose_images)
        normal_labels = [0] * len(normal_images)
        
        # Combine data
        all_images = varicose_images + normal_images
        all_labels = varicose_labels + normal_labels
        
        print(f"Dataset summary:")
        print(f"Varicose images: {len(varicose_images)}")
        print(f"Normal images: {len(normal_images)}")
        print(f"Total images: {len(all_images)}")
        print(f"Class imbalance ratio: {len(normal_images) / len(varicose_images):.2f}")
        
        # Shuffle data
        combined = list(zip(all_images, all_labels))
        random.shuffle(combined)
        all_images, all_labels = zip(*combined)
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(all_images))
        
        train_images = list(all_images[:split_idx])
        train_labels = list(all_labels[:split_idx])
        val_images = list(all_images[split_idx:])
        val_labels = list(all_labels[split_idx:])
        
        # Create datasets
        train_dataset = AdvancedVaricoseDataset(
            train_images, train_labels, is_training=True
        )
        val_dataset = AdvancedVaricoseDataset(
            val_images, val_labels, is_training=False
        )
        
        # Calculate class weights for weighted sampling
        class_counts = np.bincount(train_labels)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        
        # Create weighted sampler for balanced training
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, class_weights
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets, all_probabilities
    
    def find_optimal_threshold(self, val_loader, target_recall=0.90):
        """Find optimal threshold for high varicose recall"""
        self.model.eval()
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of varicose
                
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Find threshold that maximizes recall for varicose class
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0.0
        best_recall = 0.0
        
        results = []
        
        for threshold in thresholds:
            predictions = (np.array(all_probabilities) >= threshold).astype(int)
            
            # Calculate metrics
            recall = recall_score(all_targets, predictions)
            precision = precision_score(all_targets, predictions, zero_division=0)
            f1 = f1_score(all_targets, predictions, zero_division=0)
            accuracy = accuracy_score(all_targets, predictions)
            
            results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'accuracy': accuracy
            })
            
            # Select threshold that achieves target recall with best F1
            if recall >= target_recall and f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
                best_recall = recall
        
        print(f"Optimal threshold: {best_threshold:.3f}")
        print(f"Best varicose recall: {best_recall:.3f}")
        print(f"Best F1 score: {best_f1:.3f}")
        
        return best_threshold, results
    
    def evaluate_model(self, val_loader, threshold=0.5):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Apply custom threshold for varicose detection
                varicose_probs = probabilities[:, 1]
                predicted = (varicose_probs >= threshold).long()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average=None, zero_division=0)
        recall = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        print(f"\n{'='*50}")
        print(f"COMPREHENSIVE EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Normal Class - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
        print(f"Varicose Class - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
        print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Varicose'],
                   yticklabels=['Normal', 'Varicose'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'varicose_recall': recall[1],
            'varicose_precision': precision[1]
        }
    
    def train(self, train_loader, val_loader, class_weights):
        """Main training loop with all optimizations"""
        
        # Loss function with class weighting
        if self.config.get('use_focal_loss', True):
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            weight = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Early stopping
        patience = self.config.get('patience', 15)
        patience_counter = 0
        
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Target: 95%+ accuracy with 90%+ varicose recall")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets, val_probs = self.validate_epoch(val_loader, criterion)
            
            # Calculate varicose recall
            val_recall = recall_score(val_targets, val_preds, average=None, zero_division=0)
            varicose_recall = val_recall[1] if len(val_recall) > 1 else 0.0
            
            # Update learning rate
            scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Varicose Recall: {varicose_recall:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model based on combined metric
            combined_metric = val_acc * 0.6 + varicose_recall * 0.4  # Weight accuracy and recall
            
            if combined_metric > (self.best_val_accuracy * 0.6 + self.best_varicose_recall * 0.4):
                self.best_val_accuracy = val_acc
                self.best_varicose_recall = varicose_recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'varicose_recall': varicose_recall,
                    'threshold': 0.5
                }, 'best_varicose_model.pth')
                print(f"New best model saved! Combined metric: {combined_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            # Check if we've reached our target
            if val_acc >= 0.95 and varicose_recall >= 0.90:
                print(f"🎉 TARGET ACHIEVED! Accuracy: {val_acc:.4f}, Varicose Recall: {varicose_recall:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'varicose_recall': varicose_recall,
                }, 'target_achieved_model.pth')
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Best varicose recall: {self.best_varicose_recall:.4f}")

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        'batch_size': 16,  # Adjust based on GPU memory
        'learning_rate': 1e-4,
        'epochs': 100,
        'dropout_rate': 0.5,
        'weight_decay': 0.01,
        'patience': 15,
        'use_focal_loss': True,
    }
    
    # Data directory structure should be:
    # data/
    #   ├── varicose/
    #   │   ├── img1.jpg
    #   │   └── img2.jpg
    #   └── normal/
    #       ├── img1.jpg
    #       └── img2.jpg
    
    data_dir = "data"  # Update this path
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        print("Please create the data directory with 'varicose' and 'normal' subdirectories")
        return
    
    # Initialize trainer
    trainer = VaricoseTrainer(config)
    
    # Prepare data
    train_loader, val_loader, class_weights = trainer.prepare_data(data_dir)
    
    # Train model
    trainer.train(train_loader, val_loader, class_weights)
    
    # Load best model for evaluation
    if os.path.exists('best_varicose_model.pth'):
        checkpoint = torch.load('best_varicose_model.pth')
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("\nLoaded best model for final evaluation")
    
    # Find optimal threshold
    optimal_threshold, threshold_results = trainer.find_optimal_threshold(val_loader, target_recall=0.90)
    
    # Final evaluation with optimal threshold
    final_metrics = trainer.evaluate_model(val_loader, threshold=optimal_threshold)
    
    # Save threshold results
    pd.DataFrame(threshold_results).to_csv('threshold_analysis.csv', index=False)
    
    # Save final model with threshold
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimal_threshold': optimal_threshold,
        'final_metrics': final_metrics,
        'config': config
    }, 'final_varicose_model.pth')
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Final accuracy: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"Varicose recall: {final_metrics['varicose_recall']:.4f} ({final_metrics['varicose_recall']*100:.2f}%)")
    print(f"Varicose precision: {final_metrics['varicose_precision']:.4f}")
    
    if final_metrics['accuracy'] >= 0.95 and final_metrics['varicose_recall'] >= 0.90:
        print("🎉 SUCCESS: Target metrics achieved!")
    else:
        print("⚠️  Target not fully achieved. Consider:")
        print("- Collecting more balanced data")
        print("- Adjusting augmentation parameters")
        print("- Fine-tuning hyperparameters")

if __name__ == "__main__":
    main()
