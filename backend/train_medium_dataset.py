"""
Optimized Training for Medium Dataset (5K Images)
Expected Training Time: 2-4 hours
Target: 95%+ Overall Accuracy with 90%+ Varicose Recall

This script is optimized for medium-scale training with efficient GPU utilization.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import the advanced training components
from train_varicose_classifier import (
    VaricoseTrainer, AdvancedVaricoseDataset, EfficientNetVaricoseClassifier, 
    FocalLoss, set_seed
)

class MediumDatasetTrainer(VaricoseTrainer):
    """Optimized trainer for medium datasets (5K images)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.training_start_time = None
        self.estimated_completion = None
        self.checkpoint_interval = 5  # Save checkpoint every 5 epochs
        
        # Optimize for medium dataset
        self.setup_medium_optimizations()
        
    def setup_medium_optimizations(self):
        """Setup optimizations specific to medium datasets"""
        print("🔧 Setting up medium dataset optimizations...")
        
        # Enable mixed precision training for speed
        if torch.cuda.is_available():
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
            print("  ✅ Mixed precision training enabled (faster GPU training)")
        else:
            self.use_amp = False
            print("  ⚠️  GPU not available - training will be slower")
        
        # Set optimal number of workers based on CPU cores
        self.num_workers = min(8, os.cpu_count() or 4)
        print(f"  ✅ DataLoader workers: {self.num_workers}")
        
        # Pin memory for faster GPU transfer
        self.pin_memory = torch.cuda.is_available()
        print(f"  ✅ Pin memory: {self.pin_memory}")
    
    def prepare_data(self, data_dir):
        """Prepare data with optimizations for medium datasets"""
        print("\n📊 Preparing medium dataset...")
        
        # Call parent method
        train_loader, val_loader, class_weights = super().prepare_data(data_dir)
        
        # Optimize data loaders for medium dataset
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        # Create optimized data loaders
        train_sampler = train_loader.sampler
        
        optimized_train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,  # Prefetch batches for faster loading
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        optimized_val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        print(f"  ✅ Optimized data loaders created")
        print(f"  📊 Training batches: {len(optimized_train_loader)}")
        print(f"  📊 Validation batches: {len(optimized_val_loader)}")
        
        return optimized_train_loader, optimized_val_loader, class_weights
    
    def train_epoch_with_amp(self, train_loader, optimizer, criterion):
        """Training epoch with automatic mixed precision"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Progress reporting every 20 batches
            if batch_idx % 20 == 0:
                current_acc = 100. * correct / total
                print(f'    Batch {batch_idx:3d}/{len(train_loader):3d} | '
                      f'Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def estimate_training_time(self, train_loader, val_loader):
        """Estimate total training time based on first epoch"""
        print("\n⏱️  Estimating training time...")
        
        # Time a few batches to estimate
        self.model.train()
        batch_times = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 5:  # Test first 5 batches
                    break
                    
                start_time = time.time()
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                batch_time = time.time() - start_time
                batch_times.append(batch_time)
        
        avg_batch_time = np.mean(batch_times)
        
        # Estimate epoch time
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        
        estimated_train_time = avg_batch_time * train_batches
        estimated_val_time = avg_batch_time * val_batches * 0.5  # Validation is faster
        estimated_epoch_time = estimated_train_time + estimated_val_time
        
        # Total training time
        total_epochs = self.config['epochs']
        total_estimated_time = estimated_epoch_time * total_epochs
        
        print(f"  📊 Estimated times:")
        print(f"    - Per batch: {avg_batch_time:.3f}s")
        print(f"    - Per epoch: {estimated_epoch_time/60:.1f} minutes")
        print(f"    - Total training: {total_estimated_time/3600:.1f} hours")
        
        # Set completion time
        self.estimated_completion = datetime.now() + timedelta(seconds=total_estimated_time)
        print(f"    - Estimated completion: {self.estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return total_estimated_time
    
    def save_checkpoint(self, epoch, optimizer, val_acc, varicose_recall, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
            'varicose_recall': varicose_recall,
            'config': self.config,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        # Save regular checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, 'best_medium_model.pth')
            print(f"  💾 Best model saved (Accuracy: {val_acc:.3f}, Recall: {varicose_recall:.3f})")
        
        # Keep only last 3 checkpoints to save space
        self._cleanup_old_checkpoints(epoch)
    
    def _cleanup_old_checkpoints(self, current_epoch):
        """Remove old checkpoints to save disk space"""
        if current_epoch > 3:
            old_checkpoint = f'checkpoint_epoch_{current_epoch-3}.pth'
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def train_with_monitoring(self, train_loader, val_loader, class_weights):
        """Enhanced training with detailed monitoring"""
        
        # Setup training components
        if self.config.get('use_focal_loss', True):
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            weight = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler optimized for medium datasets
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['learning_rate'] * 10,
            steps_per_epoch=len(train_loader),
            epochs=self.config['epochs'],
            pct_start=0.3  # 30% of training for warmup
        )
        
        # Training monitoring
        self.training_start_time = time.time()
        patience = self.config.get('patience', 15)
        patience_counter = 0
        
        print(f"\n🚀 Starting optimized training for medium dataset...")
        print(f"Target: 95%+ accuracy with 90%+ varicose recall")
        print("=" * 80)
        
        # Estimate training time
        total_estimated_time = self.estimate_training_time(train_loader, val_loader)
        
        # Training loop
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Training phase with AMP
            if self.use_amp:
                train_loss, train_acc = self.train_epoch_with_amp(train_loader, optimizer, criterion)
            else:
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets, val_probs = self.validate_epoch(val_loader, criterion)
            
            # Calculate metrics
            from sklearn.metrics import recall_score, precision_score
            val_recall = recall_score(val_targets, val_preds, average=None, zero_division=0)
            varicose_recall = val_recall[1] if len(val_recall) > 1 else 0.0
            
            if len(val_recall) > 1:
                normal_recall = val_recall[0]
                val_precision = precision_score(val_targets, val_preds, average=None, zero_division=0)
                normal_precision = val_precision[0] if len(val_precision) > 0 else 0.0
                varicose_precision = val_precision[1] if len(val_precision) > 1 else 0.0
            else:
                normal_recall = 0.0
                normal_precision = 0.0
                varicose_precision = 0.0
            
            # Update learning rate
            scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Calculate epoch time and remaining time
            epoch_time = time.time() - epoch_start
            elapsed_total = time.time() - self.training_start_time
            epochs_remaining = self.config['epochs'] - (epoch + 1)
            estimated_remaining = epoch_time * epochs_remaining
            
            # Print detailed metrics
            print(f"\n📊 Results:")
            print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"  📈 Class Performance:")
            print(f"    Normal   → Recall: {normal_recall:.3f} ({normal_recall*100:.1f}%) | Precision: {normal_precision:.3f}")
            print(f"    Varicose → Recall: {varicose_recall:.3f} ({varicose_recall*100:.1f}%) | Precision: {varicose_precision:.3f}")
            print(f"  ⏱️  Time: {epoch_time:.1f}s | Remaining: {estimated_remaining/3600:.1f}h | Total: {elapsed_total/3600:.2f}h")
            print(f"  🎯 LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for target achievement
            target_achieved = val_acc >= 0.95 and varicose_recall >= 0.90
            if target_achieved:
                print(f"\n🎉 TARGET ACHIEVED!")
                print(f"  ✅ Overall Accuracy: {val_acc:.4f} (≥95%)")
                print(f"  ✅ Varicose Recall: {varicose_recall:.4f} (≥90%)")
                
                # Save target model
                self.save_checkpoint(epoch, optimizer, val_acc, varicose_recall, is_best=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                    'varicose_recall': varicose_recall,
                    'target_achieved': True,
                    'training_time': elapsed_total
                }, 'target_achieved_medium.pth')
                
                print(f"🎯 Training could stop here, but continuing for potential improvement...")
            
            # Model saving logic
            combined_metric = val_acc * 0.6 + varicose_recall * 0.4
            best_combined = self.best_val_accuracy * 0.6 + self.best_varicose_recall * 0.4
            
            is_best = combined_metric > best_combined
            if is_best:
                self.best_val_accuracy = val_acc
                self.best_varicose_recall = varicose_recall
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch, optimizer, val_acc, varicose_recall, is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping after {patience} epochs without improvement")
                break
        
        # Training completion
        total_time = time.time() - self.training_start_time
        print(f"\n✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"🕒 Total training time: {total_time/3600:.2f} hours")
        print(f"🎯 Best validation accuracy: {self.best_val_accuracy:.4f} ({self.best_val_accuracy*100:.2f}%)")
        print(f"🎯 Best varicose recall: {self.best_varicose_recall:.4f} ({self.best_varicose_recall*100:.2f}%)")
        
        # Final model evaluation
        self.final_evaluation(val_loader)
    
    def final_evaluation(self, val_loader):
        """Final model evaluation with comprehensive metrics"""
        print(f"\n📊 FINAL MODEL EVALUATION")
        print("=" * 50)
        
        # Load best model
        if os.path.exists('best_medium_model.pth'):
            checkpoint = torch.load('best_medium_model.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✅ Loaded best model for evaluation")
        
        # Find optimal threshold
        print(f"\n🎯 Finding optimal threshold for 90%+ recall...")
        optimal_threshold, threshold_results = self.find_optimal_threshold(val_loader, target_recall=0.90)
        
        # Final evaluation with optimal threshold
        final_metrics = self.evaluate_model(val_loader, threshold=optimal_threshold)
        
        # Save comprehensive results
        final_results = {
            'training_completed': datetime.now().isoformat(),
            'total_training_time_hours': (time.time() - self.training_start_time) / 3600,
            'best_val_accuracy': self.best_val_accuracy,
            'best_varicose_recall': self.best_varicose_recall,
            'optimal_threshold': optimal_threshold,
            'final_metrics': final_metrics,
            'config': self.config
        }
        
        # Save results
        with open('medium_training_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        pd.DataFrame(threshold_results).to_csv('medium_threshold_analysis.csv', index=False)
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimal_threshold': optimal_threshold,
            'final_metrics': final_metrics,
            'config': self.config,
            'training_results': final_results
        }, 'final_medium_varicose_model.pth')
        
        print(f"\n🎊 SUCCESS SUMMARY:")
        print(f"  📈 Final Accuracy: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
        print(f"  🎯 Varicose Recall: {final_metrics['varicose_recall']:.4f} ({final_metrics['varicose_recall']*100:.2f}%)")
        print(f"  💎 Varicose Precision: {final_metrics['varicose_precision']:.4f}")
        print(f"  🔧 Optimal Threshold: {optimal_threshold:.3f}")
        print(f"  ⏱️  Training Time: {(time.time() - self.training_start_time)/3600:.2f} hours")
        
        # Check if targets were met
        success = final_metrics['accuracy'] >= 0.95 and final_metrics['varicose_recall'] >= 0.90
        if success:
            print(f"\n🎉 MISSION ACCOMPLISHED!")
            print(f"   ✅ Accuracy target: {final_metrics['accuracy']*100:.1f}% ≥ 95%")
            print(f"   ✅ Recall target: {final_metrics['varicose_recall']*100:.1f}% ≥ 90%")
            print(f"   🏆 Model ready for production deployment!")
        else:
            print(f"\n⚠️  Targets not fully achieved:")
            if final_metrics['accuracy'] < 0.95:
                print(f"   📉 Accuracy: {final_metrics['accuracy']*100:.1f}% (need 95%+)")
            if final_metrics['varicose_recall'] < 0.90:
                print(f"   📉 Recall: {final_metrics['varicose_recall']*100:.1f}% (need 90%+)")
            print(f"   💡 Consider: more training data, different hyperparameters, or longer training")

def main():
    """Main function optimized for medium dataset training"""
    print("🏥 MEDIUM DATASET VARICOSE VEIN CLASSIFIER TRAINING")
    print("=" * 80)
    print("Dataset Size: 5K images")
    print("Expected Time: 2-4 hours")
    print("Target: 95%+ accuracy, 90%+ varicose recall")
    print("=" * 80)
    
    # Optimized configuration for medium dataset
    config = {
        # Core training settings
        'batch_size': 32,           # Increased for medium dataset
        'learning_rate': 1e-4,      # Conservative start
        'epochs': 50,               # Reduced epochs (early stopping will handle)
        'dropout_rate': 0.5,        # Regularization
        'weight_decay': 0.01,       # L2 regularization
        
        # Training optimizations
        'patience': 12,             # Reduced patience for faster training
        'use_focal_loss': True,     # Handle class imbalance
        
        # Medium dataset optimizations
        'warmup_epochs': 5,         # Warmup period
        'gradient_clipping': 1.0,   # Stable training
    }
    
    print(f"\n🔧 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check data availability
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"\n❌ Data directory '{data_dir}' not found!")
        print("📁 Please ensure your data is organized as:")
        print("   data/")
        print("   ├── varicose/  (varicose vein images)")
        print("   └── normal/    (normal leg images)")
        print("\n💡 Run 'python setup_classifier.py' to create the structure")
        return
    
    # Count images
    varicose_images = len(list(Path(data_dir).glob("varicose/*")))
    normal_images = len(list(Path(data_dir).glob("normal/*")))
    total_images = varicose_images + normal_images
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Varicose images: {varicose_images:,}")
    print(f"  Normal images: {normal_images:,}")
    print(f"  Total images: {total_images:,}")
    
    if total_images < 1000:
        print(f"\n⚠️  WARNING: Small dataset ({total_images} images)")
        print(f"   For optimal results, consider collecting more data")
        print(f"   Minimum recommended: 2,000 images (1,000 per class)")
    elif total_images >= 5000:
        print(f"\n✅ Excellent dataset size for high-performance training!")
    else:
        print(f"\n👍 Good dataset size - should achieve solid performance")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🚀 GPU Acceleration Available:")
        print(f"  Device: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print(f"\n⚠️  No GPU detected - training will be slower")
        print(f"   Consider using Google Colab or a GPU-enabled machine")
        print(f"   CPU training time estimate: 8-12 hours")
    
    # Initialize trainer
    print(f"\n🚀 Initializing medium dataset trainer...")
    trainer = MediumDatasetTrainer(config)
    
    # Prepare data
    print(f"\n📊 Loading and preparing dataset...")
    train_loader, val_loader, class_weights = trainer.prepare_data(data_dir)
    
    print(f"\n✅ Data preparation complete:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Class weights: Normal={class_weights[0]:.2f}, Varicose={class_weights[1]:.2f}")
    
    # Start training
    print(f"\n🏁 STARTING TRAINING...")
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        trainer.train_with_monitoring(train_loader, val_loader, class_weights)
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Total duration: {total_duration}")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        print(f"💾 Partial results saved in checkpoints")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print(f"💡 Check the error message and training logs")
    
    print(f"\n📁 Generated files:")
    print(f"  🏆 final_medium_varicose_model.pth - Production model")
    print(f"  📊 medium_training_results.json - Complete results")
    print(f"  📈 medium_threshold_analysis.csv - Threshold optimization")
    print(f"  💾 best_medium_model.pth - Best checkpoint")
    print(f"  📊 confusion_matrix.png - Visual evaluation")

if __name__ == "__main__":
    # Set seeds for reproducibility
    set_seed(42)
    
    # Run training
    main()
