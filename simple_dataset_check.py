import os
from collections import defaultdict

def check_dataset():
    """Simple dataset structure check"""
    print("🔍 Checking Dataset Structure and Accuracy...")
    
    dataset_path = "dataset"
    images_path = os.path.join(dataset_path, "images")
    annotations_path = os.path.join(dataset_path, "annotations")
    
    if not os.path.exists(images_path):
        print(f"❌ Images directory not found: {images_path}")
        return
    
    if not os.path.exists(annotations_path):
        print(f"❌ Annotations directory not found: {annotations_path}")
        return
    
    # Count files
    image_count = 0
    annotation_count = 0
    varicose_images = 0
    normal_images = 0
    matched_pairs = 0
    
    # Get all image files
    image_files = set()
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_count += 1
            base_name = filename.rsplit('.', 1)[0]
            image_files.add(base_name)
            
            if filename.startswith('nor_'):
                normal_images += 1
            else:
                varicose_images += 1
    
    # Get all annotation files
    annotation_files = set()
    for filename in os.listdir(annotations_path):
        if filename.endswith('.txt') and not filename.startswith('README'):
            annotation_count += 1
            base_name = filename.rsplit('.', 1)[0]
            annotation_files.add(base_name)
    
    # Check matched pairs
    matched_pairs = len(image_files.intersection(annotation_files))
    
    # Check annotation format (sample)
    valid_annotations = 0
    invalid_annotations = 0
    class_distribution = defaultdict(int)
    
    sample_annotations = list(annotation_files)[:10]  # Check first 10
    for base_name in sample_annotations:
        ann_file = None
        for f in os.listdir(annotations_path):
            if f.startswith(base_name) and f.endswith('.txt'):
                ann_file = f
                break
        
        if ann_file:
            ann_path = os.path.join(annotations_path, ann_file)
            try:
                with open(ann_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        lines = content.split('\\n')
                        for line in lines:
                            parts = line.split()
                            if len(parts) == 5:
                                class_id, x_center, y_center, width, height = map(float, parts)
                                class_distribution[int(class_id)] += 1
                                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 <= width <= 1 and 0 <= height <= 1):
                                    valid_annotations += 1
                                else:
                                    invalid_annotations += 1
                            else:
                                invalid_annotations += 1
            except Exception as e:
                print(f"❌ Error reading {ann_file}: {str(e)}")
                invalid_annotations += 1
    
    # Print results
    print(f"\\n📊 Dataset Analysis Results:")
    print(f"├── Total Images: {image_count}")
    print(f"├── Total Annotations: {annotation_count}")
    print(f"├── Matched Pairs: {matched_pairs}")
    print(f"├── Varicose Images: {varicose_images}")
    print(f"├── Normal Images: {normal_images}")
    print(f"├── Data Balance Ratio: {min(varicose_images, normal_images) / max(varicose_images, normal_images):.2f}")
    print(f"├── Valid Annotations (sample): {valid_annotations}")
    print(f"├── Invalid Annotations (sample): {invalid_annotations}")
    print(f"└── Class Distribution: {dict(class_distribution)}")
    
    # Dataset quality assessment
    quality_score = 0
    if matched_pairs > 0.9 * image_count:
        quality_score += 30
    elif matched_pairs > 0.8 * image_count:
        quality_score += 20
    else:
        quality_score += 10
        
    balance_ratio = min(varicose_images, normal_images) / max(varicose_images, normal_images)
    if balance_ratio > 0.8:
        quality_score += 30
    elif balance_ratio > 0.6:
        quality_score += 20
    else:
        quality_score += 10
    
    if valid_annotations > invalid_annotations:
        quality_score += 40
    elif valid_annotations == invalid_annotations:
        quality_score += 20
    
    print(f"\\n🎯 Dataset Quality Score: {quality_score}/100")
    
    if quality_score >= 80:
        print("✅ Dataset Quality: Excellent - Ready for training!")
    elif quality_score >= 60:
        print("⚠️  Dataset Quality: Good - Minor improvements recommended")
    else:
        print("❌ Dataset Quality: Needs Improvement - Address issues before training")
    
    # Show sample files for verification
    print(f"\\n📂 Sample Files:")
    sample_images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    for img in sample_images:
        print(f"├── Image: {img}")
        base_name = img.rsplit('.', 1)[0]
        ann_file = None
        for f in os.listdir(annotations_path):
            if f.startswith(base_name) and f.endswith('.txt'):
                ann_file = f
                break
        if ann_file:
            print(f"│   └── Annotation: {ann_file}")
        else:
            print(f"│   └── ❌ No matching annotation found")
    
    return quality_score

if __name__ == "__main__":
    check_dataset()
