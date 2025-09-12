import os
import cv2
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def simple_varicose_detector(image_path):
    """
    Simple varicose vein detection based on image analysis
    This is a baseline detector for testing dataset quality
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"diagnosis": "Error", "confidence": 0.0, "severity": "Unknown"}
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple vein detection based on color and texture analysis
        # Look for darker, bluish regions (typical of varicose veins)
        
        # Color analysis - look for darker blue/purple regions
        lower_blue = np.array([100, 50, 20])
        upper_blue = np.array([130, 255, 100])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Dark region detection
        dark_mask = gray < 80
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, dark_mask.astype(np.uint8) * 255)
        
        # Calculate features
        dark_ratio = np.sum(dark_mask) / (image.shape[0] * image.shape[1])
        blue_ratio = np.sum(blue_mask > 0) / (image.shape[0] * image.shape[1])
        combined_ratio = np.sum(combined_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Simple threshold-based classification
        varicose_score = (dark_ratio * 0.4 + blue_ratio * 0.6) * 100
        
        # Determine diagnosis
        if varicose_score > 15:
            diagnosis = "Varicose Veins Detected"
            if varicose_score > 30:
                severity = "Severe"
            elif varicose_score > 20:
                severity = "Moderate"
            else:
                severity = "Mild"
        else:
            diagnosis = "No Varicose Veins Detected"
            severity = "Normal"
        
        confidence = min(95, max(60, varicose_score * 3))
        
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "severity": severity,
            "varicose_score": varicose_score,
            "features": {
                "dark_ratio": dark_ratio,
                "blue_ratio": blue_ratio,
                "combined_ratio": combined_ratio
            }
        }
    
    except Exception as e:
        return {"diagnosis": "Error", "confidence": 0.0, "severity": "Unknown", "error": str(e)}

def test_model_on_dataset(sample_size=50):
    """Test the simple model on dataset samples"""
    print(f"🧪 Testing Simple Model on {sample_size} samples...")
    
    dataset_path = "dataset"
    images_path = os.path.join(dataset_path, "images")
    
    if not os.path.exists(images_path):
        print(f"❌ Images directory not found: {images_path}")
        return None
    
    # Get all image files
    all_images = []
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(filename)
    
    if len(all_images) < sample_size:
        sample_size = len(all_images)
        print(f"⚠️  Using all {sample_size} available images")
    
    # Random sample
    sample_images = random.sample(all_images, sample_size)
    
    # Test results
    results = {
        "total_tested": 0,
        "correct_predictions": 0,
        "varicose_correct": 0,
        "normal_correct": 0,
        "varicose_total": 0,
        "normal_total": 0,
        "predictions": [],
        "confidence_scores": [],
        "varicose_scores": []
    }
    
    print("\\nTesting samples:")
    print("-" * 80)
    
    for i, filename in enumerate(sample_images, 1):
        image_path = os.path.join(images_path, filename)
        
        # Ground truth based on filename
        ground_truth = "Normal" if filename.startswith('nor_') else "Varicose"
        
        # Run detection
        detection_result = simple_varicose_detector(image_path)
        
        if detection_result["diagnosis"] == "Error":
            print(f"❌ {i:2d}. {filename}: Error - {detection_result.get('error', 'Unknown error')}")
            continue
        
        predicted_class = "Normal" if "No" in detection_result["diagnosis"] else "Varicose"
        confidence = detection_result["confidence"]
        varicose_score = detection_result.get("varicose_score", 0)
        
        # Check if prediction is correct
        is_correct = (ground_truth == predicted_class)
        
        # Update results
        results["total_tested"] += 1
        results["confidence_scores"].append(confidence)
        results["varicose_scores"].append(varicose_score)
        
        if ground_truth == "Varicose":
            results["varicose_total"] += 1
            if is_correct:
                results["varicose_correct"] += 1
        else:
            results["normal_total"] += 1
            if is_correct:
                results["normal_correct"] += 1
        
        if is_correct:
            results["correct_predictions"] += 1
        
        results["predictions"].append({
            "filename": filename,
            "ground_truth": ground_truth,
            "predicted": predicted_class,
            "confidence": confidence,
            "varicose_score": varicose_score,
            "correct": is_correct
        })
        
        # Display result
        status = "✓" if is_correct else "✗"
        print(f"{status} {i:2d}. {filename[:30]:<30} | GT: {ground_truth:<8} | Pred: {predicted_class:<8} | Conf: {confidence:5.1f}% | Score: {varicose_score:5.1f}")
    
    return results

def analyze_results(results):
    """Analyze and display test results"""
    if not results or results["total_tested"] == 0:
        print("❌ No valid test results to analyze")
        return
    
    # Calculate metrics
    total = results["total_tested"]
    correct = results["correct_predictions"]
    overall_accuracy = (correct / total) * 100
    
    varicose_accuracy = (results["varicose_correct"] / results["varicose_total"]) * 100 if results["varicose_total"] > 0 else 0
    normal_accuracy = (results["normal_correct"] / results["normal_total"]) * 100 if results["normal_total"] > 0 else 0
    
    avg_confidence = np.mean(results["confidence_scores"])
    std_confidence = np.std(results["confidence_scores"])
    
    avg_varicose_score = np.mean(results["varicose_scores"])
    
    print("\\n" + "="*80)
    print("📊 MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"📈 Overall Accuracy: {overall_accuracy:.1f}% ({correct}/{total})")
    print(f"🔵 Varicose Detection: {varicose_accuracy:.1f}% ({results['varicose_correct']}/{results['varicose_total']})")
    print(f"⚪ Normal Detection: {normal_accuracy:.1f}% ({results['normal_correct']}/{results['normal_total']})")
    print(f"📊 Average Confidence: {avg_confidence:.1f}% (±{std_confidence:.1f})")
    print(f"📉 Average Varicose Score: {avg_varicose_score:.1f}")
    
    # Calculate additional metrics
    if results["varicose_total"] > 0 and results["normal_total"] > 0:
        # Sensitivity (True Positive Rate)
        sensitivity = (results["varicose_correct"] / results["varicose_total"]) * 100
        # Specificity (True Negative Rate)  
        specificity = (results["normal_correct"] / results["normal_total"]) * 100
        
        print(f"🎯 Sensitivity (TPR): {sensitivity:.1f}%")
        print(f"🎯 Specificity (TNR): {specificity:.1f}%")
        
        # F1 Score components
        if results["varicose_correct"] + (results["varicose_total"] - results["varicose_correct"]) > 0:
            precision = results["varicose_correct"] / (results["varicose_correct"] + (results["normal_total"] - results["normal_correct"]))
            recall = results["varicose_correct"] / results["varicose_total"]
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"📏 F1 Score: {f1_score:.3f}")
    
    # Performance assessment
    print("\\n🎯 PERFORMANCE ASSESSMENT:")
    if overall_accuracy >= 80:
        print("✅ Excellent performance! Model is working well with the dataset.")
    elif overall_accuracy >= 70:
        print("⚡ Good performance! Some room for improvement.")
    elif overall_accuracy >= 60:
        print("⚠️  Fair performance. Consider improving the model or dataset quality.")
    else:
        print("❌ Poor performance. Model needs significant improvement.")
    
    # Data quality insights
    print("\\n📋 DATASET QUALITY INSIGHTS:")
    balance_ratio = min(results["varicose_total"], results["normal_total"]) / max(results["varicose_total"], results["normal_total"])
    print(f"⚖️  Class Balance: {balance_ratio:.2f} (1.0 is perfect balance)")
    
    if balance_ratio < 0.7:
        print("⚠️  Dataset imbalance detected. Consider balancing the classes.")
    
    # Show challenging cases
    incorrect_predictions = [p for p in results["predictions"] if not p["correct"]]
    if incorrect_predictions:
        print(f"\\n❌ CHALLENGING CASES ({len(incorrect_predictions)} incorrect predictions):")
        for i, pred in enumerate(incorrect_predictions[:5], 1):  # Show first 5
            print(f"{i}. {pred['filename'][:40]} - GT: {pred['ground_truth']}, Pred: {pred['predicted']} (Score: {pred['varicose_score']:.1f})")
        if len(incorrect_predictions) > 5:
            print(f"   ... and {len(incorrect_predictions) - 5} more")

def create_performance_visualizations(results):
    """Create visualizations of model performance"""
    if not results or results["total_tested"] == 0:
        return
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Accuracy by class
    ax1 = axes[0, 0]
    classes = ['Varicose', 'Normal', 'Overall']
    varicose_acc = (results["varicose_correct"] / results["varicose_total"]) * 100 if results["varicose_total"] > 0 else 0
    normal_acc = (results["normal_correct"] / results["normal_total"]) * 100 if results["normal_total"] > 0 else 0
    overall_acc = (results["correct_predictions"] / results["total_tested"]) * 100
    
    accuracies = [varicose_acc, normal_acc, overall_acc]
    colors = ['#ff7f7f', '#7fbfff', '#7fff7f']
    
    bars = ax1.bar(classes, accuracies, color=colors)
    ax1.set_title('Accuracy by Class')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confidence score distribution
    ax2 = axes[0, 1]
    confidence_scores = results["confidence_scores"]
    ax2.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidence_scores):.1f}%')
    ax2.set_title('Confidence Score Distribution')
    ax2.set_xlabel('Confidence (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Varicose score by ground truth
    ax3 = axes[1, 0]
    varicose_scores_by_class = defaultdict(list)
    for pred in results["predictions"]:
        varicose_scores_by_class[pred["ground_truth"]].append(pred["varicose_score"])
    
    if len(varicose_scores_by_class) > 1:
        ax3.boxplot([varicose_scores_by_class["Varicose"], varicose_scores_by_class["Normal"]], 
                   labels=["Varicose", "Normal"])
        ax3.set_title('Varicose Score Distribution by Class')
        ax3.set_ylabel('Varicose Score')
    
    # 4. Confusion Matrix
    ax4 = axes[1, 1]
    tp = results["varicose_correct"]  # True Positives
    fn = results["varicose_total"] - results["varicose_correct"]  # False Negatives
    fp = results["normal_total"] - results["normal_correct"]  # False Positives
    tn = results["normal_correct"]  # True Negatives
    
    confusion_matrix = np.array([[tp, fn], [fp, tn]])
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Varicose', 'Predicted Normal'],
                yticklabels=['Actual Varicose', 'Actual Normal'], ax=ax4)
    ax4.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_performance_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\\n📊 Performance visualizations saved as: {filename}")
    plt.show()

def main():
    """Main function to test dataset accuracy and model performance"""
    print("🚀 Starting Dataset Accuracy and Model Performance Test")
    print("="*80)
    
    # Test model on dataset
    results = test_model_on_dataset(sample_size=50)
    
    if results:
        # Analyze results
        analyze_results(results)
        
        # Create visualizations
        try:
            create_performance_visualizations(results)
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {str(e)}")
        
        print("\\n✅ Testing completed successfully!")
        return results
    else:
        print("❌ Testing failed")
        return None

if __name__ == "__main__":
    results = main()
