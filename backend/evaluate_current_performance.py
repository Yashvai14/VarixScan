"""
Current Model Performance Evaluation
Analyzes the existing varicose vein detection system to identify performance bottlenecks
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any
import cv2
from pathlib import Path

# Import current models
try:
    from ml_model import VaricoseVeinDetector
    current_model_available = True
    print("✅ Current ml_model.py imported successfully")
except ImportError as e:
    print(f"❌ Failed to import current ml_model: {e}")
    current_model_available = False

try:
    from advanced_ml_model import advanced_detector
    advanced_model_available = True
    print("✅ Advanced ml_model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import advanced_ml_model: {e}")
    advanced_model_available = False

class ModelPerformanceEvaluator:
    """Comprehensive performance evaluator for varicose vein detection models"""
    
    def __init__(self):
        self.results = {
            'current_model': {},
            'advanced_model': {},
            'comparison': {}
        }
        self.test_images = []
        self.evaluation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def find_test_images(self, max_images=50):
        """Find test images from various sources"""
        print("\n🔍 Looking for test images...")
        
        # Check common image directories
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        search_dirs = [
            'uploads',           # FastAPI upload directory
            'test_images',       # Dedicated test directory
            'data/varicose',     # Training data (if available)
            'data/normal',       # Training data (if available)
            '.',                 # Current directory
        ]
        
        found_images = []
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"  Checking {search_dir}/...")
                for ext in image_extensions:
                    pattern = f"{search_dir}/*{ext}"
                    import glob
                    images = glob.glob(pattern)
                    for img in images[:max_images]:  # Limit per directory
                        if os.path.getsize(img) > 1000:  # At least 1KB
                            found_images.append(img)
        
        # Remove duplicates and limit total
        self.test_images = list(set(found_images))[:max_images]
        
        print(f"✅ Found {len(self.test_images)} test images")
        if len(self.test_images) == 0:
            print("⚠️  No test images found. Consider adding some images to 'test_images/' directory")
        
        return self.test_images
    
    def create_synthetic_test_images(self, num_images=10):
        """Create synthetic test images if no real images are available"""
        print(f"\n🎨 Creating {num_images} synthetic test images...")
        
        os.makedirs('synthetic_test_images', exist_ok=True)
        created_images = []
        
        for i in range(num_images):
            # Create synthetic leg image with vessel-like structures
            image = self._create_synthetic_leg_image(i)
            image_path = f"synthetic_test_images/synthetic_leg_{i:03d}.jpg"
            cv2.imwrite(image_path, image)
            created_images.append(image_path)
            self.test_images.append(image_path)
        
        print(f"✅ Created {len(created_images)} synthetic test images")
        return created_images
    
    def _create_synthetic_leg_image(self, seed):
        """Create a synthetic leg image with potential vein-like structures"""
        np.random.seed(seed)
        
        # Create base skin-colored image
        height, width = 400, 300
        
        # Base skin color (varies by seed)
        base_colors = [
            [220, 180, 140],  # Light skin
            [200, 160, 120],  # Medium skin  
            [180, 140, 100],  # Darker skin
            [160, 120, 80],   # Dark skin
        ]
        base_color = base_colors[seed % len(base_colors)]
        
        image = np.ones((height, width, 3), dtype=np.uint8) * base_color
        
        # Add some texture and variation
        noise = np.random.randint(-20, 20, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add vessel-like structures for some images
        if seed % 3 != 0:  # 2/3 of images have vessel-like structures
            num_vessels = np.random.randint(1, 4)
            for _ in range(num_vessels):
                # Random vessel path
                start_x = np.random.randint(0, width//4)
                start_y = np.random.randint(height//4, 3*height//4)
                end_x = np.random.randint(3*width//4, width)
                end_y = np.random.randint(height//4, 3*height//4)
                
                # Vessel appearance
                vessel_color = [max(0, c - np.random.randint(40, 80)) for c in base_color]
                thickness = np.random.randint(2, 6)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), vessel_color, thickness)
                
                # Add some branching
                if np.random.random() > 0.5:
                    branch_x = np.random.randint(min(start_x, end_x), max(start_x, end_x))
                    branch_y = np.random.randint(0, height)
                    cv2.line(image, (branch_x, (start_y + end_y)//2), 
                            (branch_x + np.random.randint(-50, 50), branch_y), 
                            vessel_color, max(1, thickness-1))
        
        # Add some blur to make it more realistic
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
    
    def evaluate_current_model(self):
        """Evaluate the current ml_model.py performance"""
        if not current_model_available:
            print("❌ Current model not available for evaluation")
            return {}
        
        print(f"\n📊 Evaluating Current Model Performance...")
        print("=" * 60)
        
        try:
            detector = VaricoseVeinDetector()
        except Exception as e:
            print(f"❌ Failed to initialize current model: {e}")
            return {}
        
        results = {
            'model_name': 'Current VaricoseVeinDetector',
            'predictions': [],
            'processing_times': [],
            'confidence_scores': [],
            'severity_distribution': {},
            'errors': []
        }
        
        print(f"Testing with {len(self.test_images)} images...")
        
        for i, image_path in enumerate(self.test_images):
            try:
                print(f"  Processing {i+1}/{len(self.test_images)}: {os.path.basename(image_path)}")
                
                start_time = time.time()
                result = detector.detect_veins(image_path)
                processing_time = time.time() - start_time
                
                results['predictions'].append(result)
                results['processing_times'].append(processing_time)
                results['confidence_scores'].append(result.get('confidence', 0))
                
                # Track severity distribution
                severity = result.get('severity', 'Unknown')
                results['severity_distribution'][severity] = results['severity_distribution'].get(severity, 0) + 1
                
                # Print result summary
                print(f"    Result: {result.get('diagnosis', 'Unknown')} | "
                      f"Severity: {severity} | "
                      f"Confidence: {result.get('confidence', 0):.1f}% | "
                      f"Time: {processing_time:.2f}s")
                
            except Exception as e:
                error_info = f"Error processing {image_path}: {str(e)}"
                results['errors'].append(error_info)
                print(f"    ❌ {error_info}")
        
        # Calculate summary statistics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['min_processing_time'] = np.min(results['processing_times'])
            results['max_processing_time'] = np.max(results['processing_times'])
        
        if results['confidence_scores']:
            results['avg_confidence'] = np.mean(results['confidence_scores'])
            results['min_confidence'] = np.min(results['confidence_scores'])
            results['max_confidence'] = np.max(results['confidence_scores'])
            results['confidence_std'] = np.std(results['confidence_scores'])
        
        self.results['current_model'] = results
        return results
    
    def evaluate_advanced_model(self):
        """Evaluate the advanced ml_model.py performance"""
        if not advanced_model_available:
            print("❌ Advanced model not available for evaluation")
            return {}
        
        print(f"\n📊 Evaluating Advanced Model Performance...")
        print("=" * 60)
        
        results = {
            'model_name': 'Advanced VaricoseVeinDetector',
            'predictions': [],
            'processing_times': [],
            'confidence_scores': [],
            'severity_distribution': {},
            'errors': []
        }
        
        print(f"Testing with {len(self.test_images)} images...")
        
        for i, image_path in enumerate(self.test_images):
            try:
                print(f"  Processing {i+1}/{len(self.test_images)}: {os.path.basename(image_path)}")
                
                start_time = time.time()
                result = advanced_detector.detect_varicose_veins(image_path)
                processing_time = time.time() - start_time
                
                results['predictions'].append(result)
                results['processing_times'].append(processing_time)
                results['confidence_scores'].append(result.get('confidence', 0))
                
                # Track severity distribution
                severity = result.get('severity', 'Unknown')
                results['severity_distribution'][severity] = results['severity_distribution'].get(severity, 0) + 1
                
                # Print result summary
                print(f"    Result: {result.get('diagnosis', 'Unknown')} | "
                      f"Severity: {severity} | "
                      f"Confidence: {result.get('confidence', 0):.1f}% | "
                      f"Time: {processing_time:.2f}s")
                
            except Exception as e:
                error_info = f"Error processing {image_path}: {str(e)}"
                results['errors'].append(error_info)
                print(f"    ❌ {error_info}")
        
        # Calculate summary statistics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['min_processing_time'] = np.min(results['processing_times'])
            results['max_processing_time'] = np.max(results['processing_times'])
        
        if results['confidence_scores']:
            results['avg_confidence'] = np.mean(results['confidence_scores'])
            results['min_confidence'] = np.min(results['confidence_scores'])
            results['max_confidence'] = np.max(results['confidence_scores'])
            results['confidence_std'] = np.std(results['confidence_scores'])
        
        self.results['advanced_model'] = results
        return results
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        print(f"\n📋 PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Images: {len(self.test_images)}")
        print("=" * 80)
        
        # Current Model Analysis
        current = self.results.get('current_model', {})
        advanced = self.results.get('advanced_model', {})
        
        if current:
            print(f"\n🔍 CURRENT MODEL ANALYSIS")
            print("-" * 40)
            print(f"Model: {current.get('model_name', 'Unknown')}")
            print(f"Successfully processed: {len(current.get('predictions', []))} images")
            print(f"Errors: {len(current.get('errors', []))}")
            
            if current.get('avg_confidence') is not None:
                print(f"\n📊 CONFIDENCE METRICS:")
                print(f"  Average: {current['avg_confidence']:.1f}%")
                print(f"  Range: {current['min_confidence']:.1f}% - {current['max_confidence']:.1f}%")
                print(f"  Std Dev: {current['confidence_std']:.1f}%")
                
                # Identify confidence issues
                if current['avg_confidence'] < 70:
                    print(f"  ⚠️  LOW CONFIDENCE: Average confidence below 70%")
                if current['confidence_std'] > 20:
                    print(f"  ⚠️  HIGH VARIABILITY: Confidence varies significantly")
            
            if current.get('avg_processing_time') is not None:
                print(f"\n⏱️  PERFORMANCE METRICS:")
                print(f"  Average processing time: {current['avg_processing_time']:.2f}s")
                print(f"  Range: {current['min_processing_time']:.2f}s - {current['max_processing_time']:.2f}s")
                
                if current['avg_processing_time'] > 5:
                    print(f"  ⚠️  SLOW PROCESSING: Average time exceeds 5 seconds")
            
            print(f"\n🏥 DIAGNOSIS DISTRIBUTION:")
            for severity, count in current.get('severity_distribution', {}).items():
                percentage = (count / len(self.test_images)) * 100
                print(f"  {severity}: {count} images ({percentage:.1f}%)")
        
        # Advanced Model Analysis (if available)
        if advanced:
            print(f"\n🚀 ADVANCED MODEL ANALYSIS")
            print("-" * 40)
            print(f"Model: {advanced.get('model_name', 'Unknown')}")
            print(f"Successfully processed: {len(advanced.get('predictions', []))} images")
            print(f"Errors: {len(advanced.get('errors', []))}")
            
            if advanced.get('avg_confidence') is not None:
                print(f"\n📊 CONFIDENCE METRICS:")
                print(f"  Average: {advanced['avg_confidence']:.1f}%")
                print(f"  Range: {advanced['min_confidence']:.1f}% - {advanced['max_confidence']:.1f}%")
                print(f"  Std Dev: {advanced['confidence_std']:.1f}%")
            
            if advanced.get('avg_processing_time') is not None:
                print(f"\n⏱️  PERFORMANCE METRICS:")
                print(f"  Average processing time: {advanced['avg_processing_time']:.2f}s")
                print(f"  Range: {advanced['min_processing_time']:.2f}s - {advanced['max_processing_time']:.2f}s")
            
            print(f"\n🏥 DIAGNOSIS DISTRIBUTION:")
            for severity, count in advanced.get('severity_distribution', {}).items():
                percentage = (count / len(self.test_images)) * 100
                print(f"  {severity}: {count} images ({percentage:.1f}%)")
        
        # Comparison Analysis
        if current and advanced:
            self._generate_comparison_analysis(current, advanced)
        
        # Performance Issues Analysis
        self._analyze_performance_issues()
        
        # Recommendations
        self._generate_recommendations()
    
    def _generate_comparison_analysis(self, current, advanced):
        """Generate comparison between current and advanced models"""
        print(f"\n⚖️  MODEL COMPARISON")
        print("-" * 40)
        
        if current.get('avg_confidence') and advanced.get('avg_confidence'):
            conf_diff = advanced['avg_confidence'] - current['avg_confidence']
            print(f"Confidence Improvement: {conf_diff:+.1f}%")
            
            if conf_diff > 10:
                print("  ✅ Significant confidence improvement")
            elif conf_diff > 0:
                print("  ↗️  Slight confidence improvement")
            else:
                print("  ↘️  Lower confidence with advanced model")
        
        if current.get('avg_processing_time') and advanced.get('avg_processing_time'):
            time_diff = advanced['avg_processing_time'] - current['avg_processing_time']
            print(f"Processing Time Difference: {time_diff:+.2f}s")
            
            if time_diff < -1:
                print("  ⚡ Advanced model is significantly faster")
            elif time_diff < 0:
                print("  ⚡ Advanced model is faster")
            else:
                print("  🐌 Advanced model is slower")
    
    def _analyze_performance_issues(self):
        """Analyze common performance issues"""
        print(f"\n🔍 PERFORMANCE ISSUES ANALYSIS")
        print("-" * 40)
        
        issues_found = []
        current = self.results.get('current_model', {})
        
        # Low confidence analysis
        if current.get('avg_confidence', 100) < 60:
            issues_found.append("❌ CRITICAL: Very low average confidence (<60%)")
        elif current.get('avg_confidence', 100) < 75:
            issues_found.append("⚠️  WARNING: Low average confidence (<75%)")
        
        # High confidence variability
        if current.get('confidence_std', 0) > 25:
            issues_found.append("⚠️  WARNING: High confidence variability (inconsistent predictions)")
        
        # Processing time issues
        if current.get('avg_processing_time', 0) > 10:
            issues_found.append("⚠️  WARNING: Slow processing (>10s average)")
        
        # Error rate analysis
        total_predictions = len(current.get('predictions', []))
        total_errors = len(current.get('errors', []))
        if total_predictions + total_errors > 0:
            error_rate = total_errors / (total_predictions + total_errors)
            if error_rate > 0.1:
                issues_found.append(f"❌ CRITICAL: High error rate ({error_rate*100:.1f}%)")
            elif error_rate > 0.05:
                issues_found.append(f"⚠️  WARNING: Moderate error rate ({error_rate*100:.1f}%)")
        
        # Diagnosis distribution analysis
        severity_dist = current.get('severity_distribution', {})
        total_images = sum(severity_dist.values())
        if total_images > 0:
            normal_ratio = severity_dist.get('Normal', 0) / total_images
            if normal_ratio > 0.9:
                issues_found.append("⚠️  WARNING: Model may be biased toward 'Normal' classification (>90%)")
            elif normal_ratio < 0.1:
                issues_found.append("⚠️  WARNING: Model may be over-detecting varicose veins (<10% normal)")
        
        if issues_found:
            for issue in issues_found:
                print(f"  {issue}")
        else:
            print("  ✅ No major performance issues detected")
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 40)
        
        current = self.results.get('current_model', {})
        recommendations = []
        
        # Confidence-based recommendations
        if current.get('avg_confidence', 100) < 75:
            recommendations.append("🎯 PRIORITY: Improve model confidence through:")
            recommendations.append("   - Better training data quality and quantity")
            recommendations.append("   - Advanced preprocessing and augmentation")
            recommendations.append("   - Modern CNN architectures (EfficientNet, ResNet)")
            recommendations.append("   - Proper threshold optimization")
        
        # Performance recommendations
        if current.get('avg_processing_time', 0) > 5:
            recommendations.append("⚡ OPTIMIZE: Improve processing speed through:")
            recommendations.append("   - Model optimization and quantization")
            recommendations.append("   - GPU acceleration")
            recommendations.append("   - Batch processing")
            recommendations.append("   - Image preprocessing optimization")
        
        # General improvement recommendations
        recommendations.extend([
            "",
            "🚀 NEXT STEPS FOR 95%+ ACCURACY:",
            "1. Use the provided train_varicose_classifier.py",
            "2. Collect 1000+ high-quality images per class",
            "3. Train with EfficientNet-B3 + advanced augmentation",
            "4. Optimize threshold for 90%+ varicose recall",
            "5. Deploy optimized_ml_model.py for production"
        ])
        
        for rec in recommendations:
            print(f"  {rec}")
    
    def create_visualizations(self):
        """Create performance visualization charts"""
        print(f"\n📊 Creating performance visualizations...")
        
        current = self.results.get('current_model', {})
        advanced = self.results.get('advanced_model', {})
        
        if not current.get('predictions'):
            print("⚠️  No data available for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Varicose Vein Detection Model Performance Analysis', fontsize=16)
        
        # 1. Confidence Score Distribution
        if current.get('confidence_scores'):
            axes[0, 0].hist(current['confidence_scores'], bins=20, alpha=0.7, 
                           label='Current Model', color='skyblue', edgecolor='black')
            if advanced.get('confidence_scores'):
                axes[0, 0].hist(advanced['confidence_scores'], bins=20, alpha=0.7,
                               label='Advanced Model', color='orange', edgecolor='black')
            axes[0, 0].set_xlabel('Confidence Score (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Confidence Score Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Processing Time Distribution
        if current.get('processing_times'):
            axes[0, 1].hist(current['processing_times'], bins=15, alpha=0.7,
                           label='Current Model', color='lightgreen', edgecolor='black')
            if advanced.get('processing_times'):
                axes[0, 1].hist(advanced['processing_times'], bins=15, alpha=0.7,
                               label='Advanced Model', color='red', edgecolor='black')
            axes[0, 1].set_xlabel('Processing Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Processing Time Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Severity Distribution
        severity_dist = current.get('severity_distribution', {})
        if severity_dist:
            severities = list(severity_dist.keys())
            counts = list(severity_dist.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(severities)))
            
            axes[1, 0].pie(counts, labels=severities, autopct='%1.1f%%', colors=colors)
            axes[1, 0].set_title('Current Model: Severity Distribution')
        
        # 4. Model Comparison (if both available)
        if current.get('avg_confidence') and advanced.get('avg_confidence'):
            metrics = ['Avg Confidence', 'Avg Processing Time']
            current_vals = [current.get('avg_confidence', 0), 
                           current.get('avg_processing_time', 0)]
            advanced_vals = [advanced.get('avg_confidence', 0), 
                            advanced.get('avg_processing_time', 0)]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, current_vals, width, label='Current Model', alpha=0.8)
            axes[1, 1].bar(x + width/2, advanced_vals, width, label='Advanced Model', alpha=0.8)
            axes[1, 1].set_xlabel('Metrics')
            axes[1, 1].set_ylabel('Values')
            axes[1, 1].set_title('Model Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Just show confidence scores over time
            if current.get('confidence_scores'):
                axes[1, 1].plot(current['confidence_scores'], 'o-', alpha=0.7, label='Confidence')
                axes[1, 1].set_xlabel('Image Index')
                axes[1, 1].set_ylabel('Confidence (%)')
                axes[1, 1].set_title('Confidence Scores Over Test Images')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        plot_filename = f'performance_analysis_{self.evaluation_timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved visualization: {plot_filename}")
        plt.show()
    
    def save_detailed_results(self):
        """Save detailed results to files"""
        print(f"\n💾 Saving detailed results...")
        
        # Save JSON results
        json_filename = f'performance_results_{self.evaluation_timestamp}.json'
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  ✅ Saved JSON results: {json_filename}")
        
        # Create CSV summary
        summary_data = []
        for model_key in ['current_model', 'advanced_model']:
            if model_key in self.results and self.results[model_key]:
                model_data = self.results[model_key]
                summary_data.append({
                    'Model': model_data.get('model_name', model_key),
                    'Images Processed': len(model_data.get('predictions', [])),
                    'Errors': len(model_data.get('errors', [])),
                    'Avg Confidence': model_data.get('avg_confidence', 0),
                    'Min Confidence': model_data.get('min_confidence', 0),
                    'Max Confidence': model_data.get('max_confidence', 0),
                    'Confidence StdDev': model_data.get('confidence_std', 0),
                    'Avg Processing Time': model_data.get('avg_processing_time', 0),
                    'Min Processing Time': model_data.get('min_processing_time', 0),
                    'Max Processing Time': model_data.get('max_processing_time', 0),
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_filename = f'performance_summary_{self.evaluation_timestamp}.csv'
            df.to_csv(csv_filename, index=False)
            print(f"  ✅ Saved CSV summary: {csv_filename}")
    
    def run_complete_evaluation(self):
        """Run the complete evaluation process"""
        print("🩺 VARICOSE VEIN DETECTION - PERFORMANCE EVALUATION")
        print("=" * 80)
        print(f"Starting comprehensive performance analysis...")
        
        # Step 1: Find or create test images
        self.find_test_images()
        if len(self.test_images) < 5:
            print("⚠️  Too few test images found, creating synthetic ones...")
            self.create_synthetic_test_images(10)
        
        # Step 2: Evaluate current model
        if current_model_available:
            self.evaluate_current_model()
        
        # Step 3: Evaluate advanced model (if available)
        if advanced_model_available:
            self.evaluate_advanced_model()
        
        # Step 4: Generate comprehensive report
        self.generate_performance_report()
        
        # Step 5: Create visualizations
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")
        
        # Step 6: Save results
        self.save_detailed_results()
        
        print(f"\n✅ EVALUATION COMPLETE!")
        print(f"Check the generated files for detailed analysis.")

def main():
    """Main evaluation function"""
    evaluator = ModelPerformanceEvaluator()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
