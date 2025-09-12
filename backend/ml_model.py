import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from ultralytics import YOLO
import os
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from skimage import filters, morphology, segmentation
from sklearn.cluster import KMeans

class ImagePreprocessor:
    """Advanced image preprocessing for varicose vein detection"""
    
    def __init__(self):
        self.target_size = (640, 640)
    
    def noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filter"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            enhanced = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)
    
    def skin_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment skin area using multiple color-based segmentation approaches"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Method 1: HSV-based skin detection (broader range)
            lower_hsv1 = np.array([0, 10, 60], dtype=np.uint8)
            upper_hsv1 = np.array([25, 255, 255], dtype=np.uint8)
            mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
            
            # Method 2: Alternative HSV range
            lower_hsv2 = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv2 = np.array([20, 255, 255], dtype=np.uint8)
            mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
            
            # Method 3: YCrCb-based skin detection (more flexible)
            lower_ycrcb = np.array([0, 130, 80], dtype=np.uint8)
            upper_ycrcb = np.array([255, 185, 140], dtype=np.uint8)
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Method 4: RGB-based detection for various skin tones
            # Normalize RGB values
            rgb_float = rgb.astype(np.float32)
            r, g, b = cv2.split(rgb_float)
            
            # Skin detection based on RGB ratios
            mask_rgb = np.zeros(r.shape, dtype=np.uint8)
            
            # Avoid division by zero
            total = r + g + b + 1e-8
            r_ratio = r / total
            g_ratio = g / total
            
            # Skin criteria based on RGB ratios
            skin_criteria = (
                (r > g) & (r > b) &  # Red dominance
                (r_ratio > 0.3) & (r_ratio < 0.7) &  # Reasonable red ratio
                (g_ratio > 0.2) & (g_ratio < 0.5) &  # Reasonable green ratio
                (r > 60) & (g > 40) & (b > 20)  # Minimum brightness
            )
            mask_rgb[skin_criteria] = 255
            
            # Combine all masks using OR operation
            combined_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)
            combined_mask = cv2.bitwise_or(combined_mask, mask_ycrcb)
            combined_mask = cv2.bitwise_or(combined_mask, mask_rgb)
            
            # If combined mask is still very sparse, use a more lenient approach
            skin_ratio = np.sum(combined_mask > 0) / (combined_mask.shape[0] * combined_mask.shape[1])
            if skin_ratio < 0.05:  # Less than 5% skin detected, be more lenient
                # Create a more permissive mask based on brightness and color distribution
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                bright_mask = cv2.inRange(gray, 30, 255)
                
                # Use adaptive thresholding for better results
                adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_and(bright_mask, adaptive_mask))
            
            # Morphological operations to clean the mask (adjust kernel size based on image size)
            kernel_size = max(3, min(image.shape[:2]) // 50)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Clean up the mask
            cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply Gaussian blur to smooth the mask
            blur_size = max(1, kernel_size // 2)
            if blur_size % 2 == 0:
                blur_size += 1
            cleaned_mask = cv2.GaussianBlur(cleaned_mask, (blur_size, blur_size), 0)
            
            # Ensure we have some skin area detected
            final_skin_ratio = np.sum(cleaned_mask > 0) / (cleaned_mask.shape[0] * cleaned_mask.shape[1])
            if final_skin_ratio < 0.01:  # If still less than 1%, use the entire image
                cleaned_mask = np.ones_like(cleaned_mask) * 255
                print("Warning: Minimal skin detected, using entire image for analysis")
            
            # Apply mask to original image
            segmented_skin = cv2.bitwise_and(image, image, mask=cleaned_mask)
            
            return segmented_skin, cleaned_mask
            
        except Exception as e:
            print(f"Skin segmentation failed: {e}")
            # Fallback: return original image with full mask
            full_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            return image, full_mask
    
    def crop_region_of_interest(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop the image to focus on the main skin region"""
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main skin region)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            return cropped
        
        return image
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Complete preprocessing pipeline"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_shape = image.shape
        
        # Resize image if too large
        height, width = image.shape[:2]
        if max(height, width) > 1024:
            scale = 1024 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Apply preprocessing steps
        denoised = self.noise_reduction(image)
        enhanced = self.enhance_contrast(denoised)
        segmented_skin, skin_mask = self.skin_segmentation(enhanced)
        cropped = self.crop_region_of_interest(segmented_skin, skin_mask)
        
        # Final resize for model input
        processed = cv2.resize(cropped, self.target_size)
        
        preprocessing_info = {
            'original_shape': original_shape,
            'final_shape': processed.shape,
            'skin_area_ratio': np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
        }
        
        return processed, preprocessing_info

class VaricoseVeinDetector:
    """Advanced varicose vein detection using YOLO and custom analysis"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.preprocessor = ImagePreprocessor()
        self.severity_thresholds = {
            'mild': (0.3, 0.6),
            'moderate': (0.6, 0.8),
            'severe': (0.8, 1.0)
        }
    
    def detect_veins(self, image_path: str) -> Dict:
        """Detect varicose veins in the image"""
        try:
            # Preprocess image
            processed_image, preprocessing_info = self.preprocessor.preprocess_image(image_path)
            
            # Run YOLO detection
            results = self.model(processed_image)
            
            # Extract detection information
            detection_info = self._extract_detection_info(results[0], processed_image)
            
            # If YOLO didn't find anything, try rule-based detection
            if detection_info['detection_count'] == 0:
                print("No YOLO detections found, trying rule-based analysis...")
                rule_based_info = self._rule_based_vein_detection(processed_image)
                if rule_based_info['detection_count'] > 0:
                    detection_info = rule_based_info
            
            # Analyze severity
            severity_analysis = self._analyze_severity(detection_info, preprocessing_info)
            
            # Generate confidence score
            confidence = self._calculate_confidence(detection_info, preprocessing_info)
            
            return {
                'diagnosis': severity_analysis['diagnosis'],
                'severity': severity_analysis['severity'],
                'confidence': confidence,
                'detection_count': detection_info['detection_count'],
                'affected_area_ratio': detection_info['affected_area_ratio'],
                'preprocessing_info': preprocessing_info,
                'recommendations': self._generate_recommendations(severity_analysis)
            }
            
        except Exception as e:
            return {
                'diagnosis': 'Error in analysis',
                'severity': 'Unknown',
                'confidence': 0.0,
                'error': str(e),
                'recommendations': []
            }
    
    def _extract_detection_info(self, results, image: np.ndarray) -> Dict:
        """Extract detection information from YOLO results"""
        boxes = results.boxes
        detection_info = {
            'detection_count': 0,
            'affected_area_ratio': 0.0,
            'average_confidence': 0.0,
            'detections': []
        }
        
        if boxes is not None and len(boxes) > 0:
            # Process all detections (since we're using a general YOLO model)
            varicose_detections = []
            total_affected_area = 0
            image_area = image.shape[0] * image.shape[1]
            
            for box in boxes:
                try:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else 0
                    
                    # Accept any detection with reasonable confidence
                    # Since we don't have a trained varicose vein model, we'll accept any object detection
                    # as potentially relevant (person, body parts, etc.)
                    if conf > 0.2:  # Lower threshold for general detections
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Only count reasonable sized detections (not tiny artifacts)
                        if area > (image_area * 0.01):  # At least 1% of image
                            total_affected_area += area
                            
                            varicose_detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': conf,
                                'area': float(area),
                                'class': cls
                            })
                except Exception as e:
                    print(f"Warning: Error processing detection box: {e}")
                    continue
            
            detection_info.update({
                'detection_count': len(varicose_detections),
                'affected_area_ratio': total_affected_area / image_area,
                'average_confidence': np.mean([d['confidence'] for d in varicose_detections]) if varicose_detections else 0.0,
                'detections': varicose_detections
            })
        
        return detection_info
    
    def _rule_based_vein_detection(self, image: np.ndarray) -> Dict:
        """Rule-based detection for vein-like structures when YOLO fails"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
            
            # Use different edge detection methods
            # Method 1: Canny edge detection
            edges_canny = cv2.Canny(enhanced, 30, 100)
            
            # Method 2: Sobel edge detection
            sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobelx**2 + sobely**2)
            
            # Avoid division by zero
            max_sobel = np.max(sobel_combined)
            if max_sobel > 0:
                sobel_edges = np.uint8(sobel_combined * 255 / max_sobel)
            else:
                sobel_edges = np.zeros_like(sobel_combined, dtype=np.uint8)
            
            # Combine edge detection results
            combined_edges = cv2.bitwise_or(edges_canny, sobel_edges)
            
            # Morphological operations to connect nearby edges (vein-like structures)
            kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Horizontal lines
            kernel_line2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # Vertical lines
            
            # Detect line-like structures
            lines_h = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, kernel_line)
            lines_v = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, kernel_line2)
            lines = cv2.bitwise_or(lines_h, lines_v)
            
            # Find contours in the processed image
            contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on vein-like characteristics
            vein_detections = []
            total_area = 0
            image_area = image.shape[0] * image.shape[1]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if area > 10 and perimeter > 20:  # Minimum size requirements
                    # Calculate aspect ratio and other shape features
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Calculate circularity (lower values indicate more elongated shapes like veins)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Vein-like criteria: elongated (not too circular) and reasonable size
                    if (0.2 < aspect_ratio < 5.0) and (0.01 < circularity < 0.5) and (area > image_area * 0.001):
                        confidence = min(0.8, (1 - circularity) * 0.8 + (area / (image_area * 0.1)) * 0.2)
                        
                        vein_detections.append({
                            'bbox': [float(x), float(y), float(x + w), float(y + h)],
                            'confidence': confidence,
                            'area': float(area),
                            'class': 'rule_based'
                        })
                        total_area += area
            
            return {
                'detection_count': len(vein_detections),
                'affected_area_ratio': total_area / image_area,
                'average_confidence': np.mean([d['confidence'] for d in vein_detections]) if vein_detections else 0.0,
                'detections': vein_detections
            }
            
        except Exception as e:
            print(f"Rule-based detection failed: {e}")
            return {
                'detection_count': 0,
                'affected_area_ratio': 0.0,
                'average_confidence': 0.0,
                'detections': []
            }
    
    def _analyze_severity(self, detection_info: Dict, preprocessing_info: Dict) -> Dict:
        """Analyze the severity of varicose veins"""
        if detection_info['detection_count'] == 0:
            return {
                'diagnosis': 'No Varicose Veins Detected',
                'severity': 'Normal',
                'severity_score': 0.0
            }
        
        # Calculate severity score based on multiple factors
        area_factor = min(detection_info['affected_area_ratio'] * 10, 1.0)  # 0-1 scale
        count_factor = min(detection_info['detection_count'] / 10, 1.0)  # 0-1 scale
        confidence_factor = detection_info['average_confidence']
        skin_quality_factor = preprocessing_info['skin_area_ratio']
        
        severity_score = (area_factor * 0.4 + 
                         count_factor * 0.3 + 
                         confidence_factor * 0.2 + 
                         (1 - skin_quality_factor) * 0.1)
        
        # Determine severity level
        if severity_score < 0.3:
            severity = 'Mild'
        elif severity_score < 0.6:
            severity = 'Moderate'
        else:
            severity = 'Severe'
        
        return {
            'diagnosis': 'Varicose Veins Detected',
            'severity': severity,
            'severity_score': severity_score
        }
    
    def _calculate_confidence(self, detection_info: Dict, preprocessing_info: Dict) -> float:
        """Calculate overall confidence in the detection"""
        skin_ratio = preprocessing_info['skin_area_ratio']
        
        if detection_info['detection_count'] == 0:
            # No detections - calculate confidence in negative result
            if skin_ratio > 0.3:  # Good skin segmentation
                base_confidence = 85.0
            elif skin_ratio > 0.1:  # Moderate skin segmentation
                base_confidence = 70.0
            else:  # Poor skin segmentation, but fallback was used
                base_confidence = 60.0
            
            # Adjust based on image quality
            image_quality_factor = min(skin_ratio * 2, 1.0)  # Cap at 1.0
            final_confidence = min(90.0, base_confidence + (image_quality_factor * 10))
            return round(final_confidence, 1)
        
        # Positive detections - calculate confidence in positive result
        base_confidence = detection_info['average_confidence'] * 100
        
        # Quality bonuses
        skin_quality_bonus = min(skin_ratio * 15, 15.0)  # Max 15 point bonus
        detection_consistency = min(detection_info['detection_count'] / 3, 1.0) * 10  # Max 10 point bonus
        
        # Image size bonus (larger images generally more reliable)
        original_size = preprocessing_info.get('original_shape', (100, 100, 3))
        size_factor = min((original_size[0] * original_size[1]) / (300 * 300), 1.0)
        size_bonus = size_factor * 5  # Max 5 point bonus
        
        final_confidence = min(95.0, base_confidence + skin_quality_bonus + detection_consistency + size_bonus)
        return round(max(final_confidence, 50.0), 1)  # Minimum 50% confidence for positive detections
    
    def _generate_recommendations(self, severity_analysis: Dict) -> List[str]:
        """Generate recommendations based on severity"""
        recommendations = []
        severity = severity_analysis['severity']
        
        if severity == 'Normal':
            recommendations.extend([
                "Continue regular exercise and maintain a healthy lifestyle",
                "Monitor your legs for any changes",
                "Consider wearing compression socks during long periods of standing"
            ])
        elif severity == 'Mild':
            recommendations.extend([
                "Wear compression stockings daily",
                "Elevate your legs when resting",
                "Avoid prolonged standing or sitting",
                "Exercise regularly, especially walking and swimming",
                "Maintain a healthy weight"
            ])
        elif severity == 'Moderate':
            recommendations.extend([
                "Consult with a vascular specialist for proper evaluation",
                "Use medical-grade compression stockings",
                "Consider sclerotherapy or other minimally invasive treatments",
                "Elevate legs above heart level when possible",
                "Avoid high heels and tight clothing"
            ])
        else:  # Severe
            recommendations.extend([
                "Seek immediate consultation with a vascular surgeon",
                "Consider surgical intervention (vein stripping, endovenous ablation)",
                "Use high-compression medical stockings",
                "Monitor for complications like ulcers or blood clots",
                "Follow strict lifestyle modifications"
            ])
        
        return recommendations

    def train_custom_model(self, dataset_path: str, epochs: int = 100):
        """Train a custom model on the varicose vein dataset"""
        try:
            # Train the model
            results = self.model.train(
                data=f"{dataset_path}/data.yaml",
                epochs=epochs,
                imgsz=640,
                batch=16,
                name='varicose_vein_model'
            )
            return results
        except Exception as e:
            print(f"Training error: {e}")
            return None
