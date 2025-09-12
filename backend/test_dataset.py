import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime
from ml_model import VaricoseVeinDetector, ImagePreprocessor
import random

class DatasetValidator:
    """Validate and test the varicose vein dataset"""
    
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, "images")
        self.annotations_path = os.path.join(dataset_path, "annotations")
        self.detector = VaricoseVeinDetector()
        self.preprocessor = ImagePreprocessor()
        
    def validate_dataset_structure(self):
        """Validate the dataset structure and integrity"""
        print("🔍 Validating Dataset Structure...")
        
        validation_results = {
            "total_images": 0,
            "total_annotations": 0,
            "matched_pairs": 0,
            "varicose_images": 0,
            "normal_images": 0,
            "orphaned_images": [],
            "orphaned_annotations": [],
            "image_formats": defaultdict(int),
            "annotation_issues": []
        }
        
        # Get all images and annotations
        if not os.path.exists(self.images_path):
            print(f"❌ Images directory not found: {self.images_path}")
            return validation_results
            
        if not os.path.exists(self.annotations_path):
            print(f"❌ Annotations directory not found: {self.annotations_path}")
            return validation_results
        
        # Scan images
        image_files = {}
        for filename in os.listdir(self.images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                validation_results["total_images"] += 1
                ext = filename.split('.')[-1].lower()
                validation_results["image_formats"][ext] += 1
                
                # Get base name for matching with annotations
                base_name = filename.rsplit('.', 1)[0]
                image_files[base_name] = filename
                
                if filename.startswith('nor_'):
                    validation_results["normal_images"] += 1
                else:
                    validation_results["varicose_images"] += 1
        
        # Scan annotations
        annotation_files = {}
        for filename in os.listdir(self.annotations_path):
            if filename.endswith('.txt') and not filename.startswith('README'):
                validation_results["total_annotations"] += 1
                base_name = filename.rsplit('.', 1)[0]
                annotation_files[base_name] = filename
        
        # Check for matched pairs
        for base_name in image_files:
            if base_name in annotation_files:
                validation_results["matched_pairs"] += 1
            else:
                validation_results["orphaned_images"].append(image_files[base_name])
        
        for base_name in annotation_files:
            if base_name not in image_files:
                validation_results["orphaned_annotations"].append(annotation_files[base_name])
        
        # Validate annotation format
        sample_annotations = list(annotation_files.keys())[:10]  # Check first 10
        for base_name in sample_annotations:
            ann_path = os.path.join(self.annotations_path, annotation_files[base_name])
            try:
                with open(ann_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        lines = content.split('\n')
                        for line in lines:
                            parts = line.split()
                            if len(parts) != 5:
                                validation_results["annotation_issues"].append(
                                    f"Invalid format in {annotation_files[base_name]}: {line}"
                                )
                            else:
                                # Check if values are in valid range
                                class_id, x_center, y_center, width, height = map(float, parts)
                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                       0 <= width <= 1 and 0 <= height <= 1):
                                    validation_results["annotation_issues"].append(
                                        f"Invalid coordinates in {annotation_files[base_name]}: {line}"
                                    )
            except Exception as e:
                validation_results["annotation_issues"].append(
                    f"Error reading {annotation_files[base_name]}: {str(e)}"
                )
        
        self._print_validation_results(validation_results)
        return validation_results
    
    def _print_validation_results(self, results):
        """Print validation results in a formatted way"""
        print("\n📊 Dataset Validation Results:")
        print(f"├── Total Images: {results['total_images']}")
        print(f"├── Total Annotations: {results['total_annotations']}")
        print(f"├── Matched Pairs: {results['matched_pairs']}")
        print(f"├── Varicose Images: {results['varicose_images']}")
        print(f"├── Normal Images: {results['normal_images']}")
        print(f"├── Orphaned Images: {len(results['orphaned_images'])}")
        print(f"├── Orphaned Annotations: {len(results['orphaned_annotations'])}")
        print(f"└── Annotation Issues: {len(results['annotation_issues'])}")
        
        if results['annotation_issues']:
            print("\n⚠️  Annotation Issues Found:")
            for issue in results['annotation_issues'][:5]:  # Show first 5
                print(f"   - {issue}")
            if len(results['annotation_issues']) > 5:
                print(f"   ... and {len(results['annotation_issues']) - 5} more")
    
    def test_model_accuracy(self, sample_size=20):
        """Test model accuracy on a sample of the dataset"""
        print(f"\n🧪 Testing Model Accuracy on {sample_size} samples...")
        
        # Get sample images
        all_images = []
        for filename in os.listdir(self.images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(filename)
        
        if len(all_images) < sample_size:
            sample_size = len(all_images)
            print(f"⚠️  Reducing sample size to {sample_size} (all available images)")
        
        sample_images = random.sample(all_images, sample_size)
        
        results = {
            "total_tested": 0,
            "correct_predictions": 0,
            "varicose_correct": 0,
            "normal_correct": 0,
            "varicose_total": 0,
            "normal_total": 0,
            "predictions": [],
            "confidence_scores": []
        }
        
        for filename in sample_images:
            image_path = os.path.join(self.images_path, filename)
            
            # Determine ground truth
            ground_truth = "Normal" if filename.startswith('nor_') else "Varicose"
            
            try:
                # Run detection
                detection_result = self.detector.detect_veins(image_path)
                predicted_class = "Normal" if "No" in detection_result.get('diagnosis', '') else "Varicose"
                confidence = detection_result.get('confidence', 0)
                
                # Record results
                is_correct = (ground_truth == predicted_class)
                results["total_tested"] += 1
                results["confidence_scores"].append(confidence)
                
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
                    "correct": is_correct
                })
                
                print(f"✓ {filename}: {ground_truth} -> {predicted_class} ({confidence:.1f}%) {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                continue
        
        # Calculate metrics
        overall_accuracy = (results["correct_predictions"] / results["total_tested"]) * 100 if results["total_tested"] > 0 else 0
        varicose_accuracy = (results["varicose_correct"] / results["varicose_total"]) * 100 if results["varicose_total"] > 0 else 0
        normal_accuracy = (results["normal_correct"] / results["normal_total"]) * 100 if results["normal_total"] > 0 else 0
        avg_confidence = np.mean(results["confidence_scores"]) if results["confidence_scores"] else 0
        
        print(f"\n📊 Accuracy Results:")
        print(f"├── Overall Accuracy: {overall_accuracy:.1f}% ({results['correct_predictions']}/{results['total_tested']})")
        print(f"├── Varicose Detection: {varicose_accuracy:.1f}% ({results['varicose_correct']}/{results['varicose_total']})")
        print(f"├── Normal Detection: {normal_accuracy:.1f}% ({results['normal_correct']}/{results['normal_total']})")
        print(f"└── Average Confidence: {avg_confidence:.1f}%")
        
        return results
    
    def analyze_image_quality(self, sample_size=10):
        """Analyze image quality and preprocessing results"""
        print(f"\n🖼️  Analyzing Image Quality on {sample_size} samples...")
        
        all_images = []
        for filename in os.listdir(self.images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(filename)
        
        sample_images = random.sample(all_images, min(sample_size, len(all_images)))
        
        quality_metrics = {
            "resolutions": [],
            "file_sizes": [],
            "skin_area_ratios": [],
            "preprocessing_success": 0,
            "preprocessing_failures": 0
        }
        
        for filename in sample_images:
            image_path = os.path.join(self.images_path, filename)
            
            try:
                # Get basic image info
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                file_size = os.path.getsize(image_path) / 1024  # KB
                
                quality_metrics["resolutions"].append((width, height))
                quality_metrics["file_sizes"].append(file_size)
                
                # Test preprocessing
                processed_image, preprocessing_info = self.preprocessor.preprocess_image(image_path)
                skin_area_ratio = preprocessing_info.get('skin_area_ratio', 0)
                
                quality_metrics["skin_area_ratios"].append(skin_area_ratio)
                quality_metrics["preprocessing_success"] += 1
                
                print(f"✓ {filename}: {width}x{height}, {file_size:.1f}KB, Skin: {skin_area_ratio:.1%}")
                
            except Exception as e:
                quality_metrics["preprocessing_failures"] += 1
                print(f"❌ {filename}: Preprocessing failed - {str(e)}")
        
        # Summary statistics
        if quality_metrics["resolutions"]:
            avg_width = np.mean([r[0] for r in quality_metrics["resolutions"]])
            avg_height = np.mean([r[1] for r in quality_metrics["resolutions"]])
            avg_file_size = np.mean(quality_metrics["file_sizes"])
            avg_skin_ratio = np.mean(quality_metrics["skin_area_ratios"])
            
            print(f"\n📈 Quality Metrics Summary:")
            print(f"├── Average Resolution: {avg_width:.0f}x{avg_height:.0f}")
            print(f"├── Average File Size: {avg_file_size:.1f}KB")
            print(f"├── Average Skin Ratio: {avg_skin_ratio:.1%}")
            print(f"├── Preprocessing Success: {quality_metrics['preprocessing_success']}")
            print(f"└── Preprocessing Failures: {quality_metrics['preprocessing_failures']}")
        
        return quality_metrics
    
    def create_data_distribution_chart(self):
        """Create a chart showing data distribution"""
        print("\n📊 Creating data distribution chart...")
        
        varicose_count = 0
        normal_count = 0
        
        for filename in os.listdir(self.images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                if filename.startswith('nor_'):
                    normal_count += 1
                else:
                    varicose_count += 1
        
        # Create pie chart
        labels = ['Varicose Veins', 'Normal']
        sizes = [varicose_count, normal_count]
        colors = ['#ff9999', '#66b3ff']
        
        plt.figure(figsize=(10, 6))
        
        # Pie chart
        plt.subplot(1, 2, 1)
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Dataset Distribution')
        
        # Bar chart
        plt.subplot(1, 2, 2)
        plt.bar(labels, sizes, color=colors)
        plt.title('Sample Counts')
        plt.ylabel('Number of Images')
        
        for i, v in enumerate(sizes):
            plt.text(i, v + max(sizes) * 0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"dataset_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved as: {chart_filename}")
        plt.show()
        
        return {
            "varicose_count": varicose_count,
            "normal_count": normal_count,
            "total": varicose_count + normal_count,
            "balance_ratio": min(varicose_count, normal_count) / max(varicose_count, normal_count)
        }
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive dataset analysis report"""
        print("🔍 Generating Comprehensive Dataset Report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": self.dataset_path,
            "validation": self.validate_dataset_structure(),
            "accuracy": self.test_model_accuracy(sample_size=30),
            "quality": self.analyze_image_quality(sample_size=15),
            "distribution": self.create_data_distribution_chart()
        }
        
        # Save report
        report_filename = f"dataset_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Comprehensive report saved as: {report_filename}")
        
        # Print summary
        print(f"\n🎯 DATASET ANALYSIS SUMMARY:")
        print(f"├── Total Images: {report['validation']['total_images']}")
        print(f"├── Data Balance: {report['distribution']['balance_ratio']:.2f}")
        print(f"├── Model Accuracy: {report['accuracy']['correct_predictions']}/{report['accuracy']['total_tested']} ({(report['accuracy']['correct_predictions']/report['accuracy']['total_tested']*100):.1f}%)")
        print(f"├── Preprocessing Success: {report['quality']['preprocessing_success']}/{report['quality']['preprocessing_success'] + report['quality']['preprocessing_failures']}")
        print(f"└── Dataset Quality: {'Good' if report['validation']['matched_pairs'] > 0.8 * report['validation']['total_images'] else 'Needs Improvement'}")
        
        return report

def main():
    """Main function to run dataset validation and testing"""
    print("🚀 Starting Varicose Vein Dataset Validation and Testing")
    print("=" * 60)
    
    validator = DatasetValidator()
    
    try:
        # Generate comprehensive report
        report = validator.generate_comprehensive_report()
        
        print("\n✅ Dataset analysis completed successfully!")
        print("Check the generated files for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()
