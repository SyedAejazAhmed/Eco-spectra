"""
Fine-tuned Solar Panel Detection Model
Training pipeline for Mask R-CNN specifically for solar panels
Eliminates false positives from shadows, buildings, and other objects
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F

# Configuration
CONFIG = {
    # Paths
    'base_dir': Path(__file__).parent,
    'data_dir': Path(__file__).parent.parent.parent / "Data Analytics" / "Google_MapStaticAPI" / "images",
    'csv_path': Path(__file__).parent.parent.parent / "Data Analytics" / "EI_train_data(Sheet1).csv",
    'output_dir': Path(__file__).parent / "finetuned_model",
    'weights_dir': Path(__file__).parent / "finetuned_model" / "weights",
    'logs_dir': Path(__file__).parent / "finetuned_model" / "logs",
    
    # Training parameters
    'num_epochs': 25,
    'batch_size': 4,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'lr_scheduler_step': 5,
    'lr_scheduler_gamma': 0.5,
    
    # Model parameters
    'num_classes': 2,  # Background + Solar Panel
    'min_size': 640,
    'max_size': 640,
    'rpn_nms_thresh': 0.7,
    'box_score_thresh': 0.5,
    'box_nms_thresh': 0.5,
    'box_detections_per_img': 100,
    
    # Data augmentation
    'use_augmentation': True,
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.3,
    'rotation_prob': 0.4,
    'brightness_factor': 0.2,
    'contrast_factor': 0.2,
    
    # Validation split
    'val_split': 0.2,
    'random_seed': 42,
    
    # Device
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

class SolarPanelDataset(Dataset):
    """
    Custom dataset for solar panel detection
    Generates pseudo-masks from existing detection results for fine-tuning
    """
    
    def __init__(self, image_paths, labels, transforms=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        
        has_solar = self.labels[idx]
        
        # Generate target
        target = {}
        
        if has_solar == 1:
            # Use enhanced detector to generate training masks
            masks, boxes = self._generate_solar_masks(img_array)
            
            if len(masks) == 0:
                # If no detection, treat as background
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
                target['masks'] = torch.zeros((0, img_array.shape[0], img_array.shape[1]), dtype=torch.uint8)
            else:
                target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)
                target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            # No solar panels - background only
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, img_array.shape[0], img_array.shape[1]), dtype=torch.uint8)
        
        target['image_id'] = torch.tensor([idx])
        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        target['iscrowd'] = torch.zeros((len(target['boxes']),), dtype=torch.int64)
        
        # Apply augmentation
        if self.augment:
            img, target = self._apply_augmentation(img, target)
        
        # Convert to tensor
        img = F.to_tensor(img)
        
        return img, target
    
    def _generate_solar_masks(self, img_array):
        """
        Generate training masks using multi-method detection
        Only returns high-confidence detections suitable for training
        """
        masks = []
        boxes = []
        
        # Method 1: Color-based detection (blue/black panels)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Blue panels (crystalline)
        blue_lower = np.array([90, 40, 40])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Black panels (monocrystalline)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 70])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(blue_mask, black_mask)
        
        # Method 2: Edge detection for rectangular structures
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine with color mask
        combined = cv2.bitwise_and(color_mask, edges)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (solar panels should be 50-5000 pixels)
            if area < 50 or area > 5000:
                continue
            
            # Filter by aspect ratio (panels are roughly rectangular)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Filter by solidity (how "filled" the shape is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.6:  # Solar panels should be fairly solid shapes
                continue
            
            # Create binary mask for this detection
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            box = [x, y, x + w, y + h]
            
            masks.append(mask)
            boxes.append(box)
        
        if len(masks) == 0:
            return np.array([]), np.array([])
        
        return np.array(masks), np.array(boxes)
    
    def _apply_augmentation(self, img, target):
        """Apply data augmentation"""
        img_array = np.array(img)
        
        # Horizontal flip
        if random.random() < CONFIG['horizontal_flip_prob']:
            img_array = np.fliplr(img_array).copy()  # .copy() to fix negative strides
            if len(target['boxes']) > 0:
                boxes = target['boxes'].numpy()
                boxes[:, [0, 2]] = img_array.shape[1] - boxes[:, [2, 0]]
                target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                
                masks = target['masks'].numpy()
                masks = np.flip(masks, axis=2).copy()  # .copy() to fix negative strides
                target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        
        # Vertical flip
        if random.random() < CONFIG['vertical_flip_prob']:
            img_array = np.flipud(img_array).copy()  # .copy() to fix negative strides
            if len(target['boxes']) > 0:
                boxes = target['boxes'].numpy()
                boxes[:, [1, 3]] = img_array.shape[0] - boxes[:, [3, 1]]
                target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                
                masks = target['masks'].numpy()
                masks = np.flip(masks, axis=1).copy()  # .copy() to fix negative strides
                target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        
        # Brightness/Contrast adjustment
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-CONFIG['contrast_factor'], CONFIG['contrast_factor'])
            beta = random.uniform(-CONFIG['brightness_factor'], CONFIG['brightness_factor']) * 255
            img_array = np.clip(alpha * img_array + beta, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        return img, target

def collate_fn(batch):
    """Custom collate function for dataloader"""
    return tuple(zip(*batch))

def get_model(num_classes):
    """
    Get Mask R-CNN model with ResNet50-FPN v2 backbone
    Improved version with better feature pyramid network
    """
    # Load pretrained model
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    num_batches = 0
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
    print(f"{'='*60}")
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        num_batches += 1
        
        if (i + 1) % 10 == 0:
            avg_loss = running_loss / num_batches
            print(f"  Batch [{i+1}/{len(data_loader)}] - Loss: {avg_loss:.4f}")
            print(f"    Breakdown: {', '.join([f'{k}: {v.item():.4f}' for k, v in loss_dict.items()])}")
    
    epoch_loss = running_loss / num_batches
    print(f"\n✅ Epoch {epoch + 1} completed - Average Loss: {epoch_loss:.4f}")
    
    return epoch_loss

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    
    total_images = 0
    total_detections = 0
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        
        outputs = model(images)
        
        for output in outputs:
            # Count detections with score > 0.5
            high_conf = output['scores'] > 0.5
            total_detections += high_conf.sum().item()
        
        total_images += len(images)
    
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    print(f"\n📊 Validation Results:")
    print(f"   Images: {total_images}")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg detections/image: {avg_detections:.2f}")
    
    return avg_detections

def prepare_training_data():
    """Prepare training and validation datasets"""
    print("📂 Preparing training data...")
    
    # Load CSV
    df = pd.read_csv(CONFIG['csv_path'], dtype={'sampleid': str})
    print(f"   Loaded {len(df)} records from CSV")
    
    # Get image paths
    image_dir = CONFIG['data_dir']
    all_images = []
    all_labels = []
    
    for _, row in df.iterrows():
        sampleid = str(row['sampleid']).zfill(4)
        has_solar = int(row['has_solar'])
        
        # Try multiple naming patterns
        patterns = [
            f"{sampleid}.png",
            f"{sampleid}_1.png",
            f"{float(sampleid)}_1.0.png",
        ]
        
        for pattern in patterns:
            img_path = image_dir / pattern
            if img_path.exists():
                all_images.append(img_path)
                all_labels.append(has_solar)
                break
    
    print(f"   Found {len(all_images)} images")
    
    # Split into train/val
    indices = list(range(len(all_images)))
    random.seed(CONFIG['random_seed'])
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - CONFIG['val_split']))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_images = [all_images[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_images = [all_images[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"   Training set: {len(train_images)} images")
    print(f"   Validation set: {len(val_images)} images")
    
    # Calculate class distribution
    train_solar = sum(train_labels)
    val_solar = sum(val_labels)
    
    print(f"\n   Training - Solar: {train_solar}, No Solar: {len(train_labels) - train_solar}")
    print(f"   Validation - Solar: {val_solar}, No Solar: {len(val_labels) - val_solar}")
    
    return train_images, train_labels, val_images, val_labels

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("🚀 SOLAR PANEL DETECTION - FINE-TUNING PIPELINE")
    print("="*80)
    
    # Create directories
    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['weights_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['logs_dir'].mkdir(parents=True, exist_ok=True)
    
    # Device info
    print(f"\n🖥️  Device: {CONFIG['device']}")
    if CONFIG['device'].type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Prepare data
    train_images, train_labels, val_images, val_labels = prepare_training_data()
    
    # Create datasets
    print("\n🔄 Creating datasets...")
    train_dataset = SolarPanelDataset(
        train_images, 
        train_labels,
        augment=CONFIG['use_augmentation']
    )
    val_dataset = SolarPanelDataset(
        val_images,
        val_labels,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n🤖 Initializing model...")
    model = get_model(CONFIG['num_classes'])
    model.to(CONFIG['device'])
    print("   Model: Mask R-CNN ResNet50-FPN v2")
    print(f"   Classes: {CONFIG['num_classes']} (Background + Solar Panel)")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['lr_scheduler_step'],
        gamma=CONFIG['lr_scheduler_gamma']
    )
    
    print(f"   Optimizer: SGD (lr={CONFIG['learning_rate']}, momentum={CONFIG['momentum']})")
    print(f"   LR Scheduler: StepLR (step={CONFIG['lr_scheduler_step']}, gamma={CONFIG['lr_scheduler_gamma']})")
    
    # Training loop
    print("\n" + "="*80)
    print("🎯 STARTING TRAINING")
    print("="*80)
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, CONFIG['device'], epoch
        )
        
        # Validate
        if (epoch + 1) % 5 == 0:
            avg_detections = evaluate(model, val_loader, CONFIG['device'])
        else:
            avg_detections = 0
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = CONFIG['weights_dir'] / f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"   💾 Saved best model checkpoint: {checkpoint_path.name}")
        
        # Log results
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_avg_detections': avg_detections,
            'learning_rate': current_lr
        })
    
    # Save final model
    final_model_path = CONFIG['weights_dir'] / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n💾 Saved final model: {final_model_path}")
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_csv = CONFIG['logs_dir'] / "training_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"📊 Saved training history: {history_csv}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(CONFIG['logs_dir'] / 'training_curve.png', dpi=150, bbox_inches='tight')
    print(f"📈 Saved training curve: {CONFIG['logs_dir'] / 'training_curve.png'}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"📁 Model saved to: {CONFIG['weights_dir']}")
    print(f"📊 Logs saved to: {CONFIG['logs_dir']}")
    print(f"🏆 Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
