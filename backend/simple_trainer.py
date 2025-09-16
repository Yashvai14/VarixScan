#!/usr/bin/env python3
"""
Simple Local Varicose Vein Trainer
A working trainer that avoids dependency conflicts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

class SimpleVaricoseNet(nn.Module):
    """Simple CNN for varicose vein classification"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleDataset(Dataset):
    """Simple dataset for varicose images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a default black image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def load_data():
    """Load dataset from data directories"""
    
    # Get image paths
    varicose_paths = []
    normal_paths = []
    
    # Check for different image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for ext in extensions:
        varicose_paths.extend(glob.glob(f'data/varicose/{ext}'))
        normal_paths.extend(glob.glob(f'data/normal/{ext}'))
    
    print(f"📊 Found images:")
    print(f"  Varicose: {len(varicose_paths)}")
    print(f"  Normal: {len(normal_paths)}")
    print(f"  Total: {len(varicose_paths) + len(normal_paths)}")
    
    if len(varicose_paths) == 0 or len(normal_paths) == 0:
        print("❌ No images found! Please add images to data/varicose/ and data/normal/")
        return None, None, None, None
    
    # Create labels (1 for varicose, 0 for normal)
    all_paths = varicose_paths + normal_paths
    all_labels = [1] * len(varicose_paths) + [0] * len(normal_paths)
    
    # Split dataset
    if len(all_paths) < 4:  # Need at least 4 images to split
        print("⚠️  Dataset too small for train/validation split")
        return all_paths, all_paths, all_labels, all_labels
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    return train_paths, val_paths, train_labels, val_labels

def train_simple_model():
    """Train the simple varicose model"""
    
    print("🏥 SIMPLE VARICOSE VEIN TRAINER")
    print("=" * 50)
    
    # Load data
    train_paths, val_paths, train_labels, val_labels = load_data()
    
    if train_paths is None:
        return False
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SimpleDataset(train_paths, train_labels, train_transform)
    val_dataset = SimpleDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    batch_size = 4  # Small batch size for compatibility
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    model = SimpleVaricoseNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 20  # Shorter training for quick test
    
    print(f"🚀 Starting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_varicose_recall': []
    }
    
    best_accuracy = 0.0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f'\rEpoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}', end='')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_varicose_recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_accuracy'].append(val_accuracy)
        history['val_varicose_recall'].append(val_varicose_recall)
        
        # Print results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'  Val Accuracy: {val_accuracy:.4f}')
        print(f'  Val Varicose Recall: {val_varicose_recall:.4f}')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_varicose_recall': val_varicose_recall
            }, 'simple_varicose_model.pth')
            print(f'  🏆 New best model saved! (Accuracy: {val_accuracy:.4f})')
    
    training_time = time.time() - start_time
    print(f'\n✅ Training completed in {training_time/60:.1f} minutes')
    print(f'🏆 Best accuracy: {best_accuracy:.4f}')
    
    # Save training history
    with open('simple_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n📁 Generated files:')
    print(f'  🏆 simple_varicose_model.pth - Best model')
    print(f'  📊 simple_training_history.json - Training history')
    
    return True

def main():
    """Main training function"""
    
    # Check if data exists
    if not os.path.exists('data'):
        print("❌ No data directory found!")
        print("Please create data/varicose/ and data/normal/ directories")
        print("And add your medical images")
        return
    
    success = train_simple_model()
    
    if success:
        print(f'\n🎉 Training completed successfully!')
        print(f'📊 This is a simple test - for 95%+ accuracy use Google Colab')
        print(f'🚀 Next step: Upload Varicose_Vein_Training_Colab.ipynb to Colab')
    else:
        print(f'\n❌ Training failed - check your data directory')

if __name__ == "__main__":
    main()
