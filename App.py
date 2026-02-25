import os
import cv2
import numpy as np
import shutil
import time
import json
import base64
import pickle
import pandas as pd
from datetime import datetime

# Suppress any verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to import Flask with better error handling
try:
    from flask import Flask, render_template, request, send_from_directory, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Flask import error: {e}")
    print("Please install Flask: pip install flask")
    FLASK_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ PyTorch import error: {e}")
    print("Please install PyTorch: pip install torch torchvision")
    TORCH_AVAILABLE = False

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"❌ YOLO import error: {e}")
    print("Please install ultralytics: pip install ultralytics")
    YOLO_AVAILABLE = False

# Try to import Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Tesseract import error: {e}")
    print("Please install pytesseract: pip install pytesseract")
    TESSERACT_AVAILABLE = False
    pytesseract = None

# Try to import EasyOCR for faster alternative
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError as e:
    print(f"❌ EasyOCR import error: {e}")
    print("Please install easyocr: pip install easyocr")
    EASYOCR_AVAILABLE = False
    easyocr = None

# Custom secure_filename implementation
try:
    from werkzeug.utils import secure_filename
except Exception:
    import re
    def secure_filename(filename):
        if not filename:
            return filename
        filename = filename.strip().replace(' ', '_')
        filename = re.sub(r'[^A-Za-z0-9._-]', '_', filename)
        return filename

# ----------------------------
# 🔹 Simple RRDBNet Architecture for Real-ESRGAN
# ----------------------------

if TORCH_AVAILABLE:
    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super(ResidualDenseBlock, self).__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super(RRDB, self).__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class SimpleRRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
            super(SimpleRRDBNet, self).__init__()
            self.num_feat = num_feat
            self.num_block = num_block
            
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            
            # Create RRDB blocks based on the actual weight structure
            self.body = nn.ModuleList()
            for i in range(num_block):
                self.body.append(RRDB(num_feat, num_grow_ch))
            
            # Note: The weights show "conv_body" not "trunk_conv"
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            
            # Upsampling layers - matches the weight names
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            # Use conv_first (matches weight names)
            fea = self.conv_first(x)
            
            # Process through RRDB blocks
            for i in range(self.num_block):
                fea = self.body[i](fea)
            
            # Use conv_body (matches weight names)
            trunk = self.conv_body(fea)
            fea = fea + trunk
            
            # Upsampling
            fea = self.lrelu(self.conv_up1(fea))
            fea = self.lrelu(self.conv_up2(fea))
            out = self.conv_last(self.lrelu(self.conv_hr(fea)))
            
            return out
else:
    # Dummy classes if PyTorch is not available
    class SimpleRRDBNet:
        def __init__(self, *args, **kwargs):
            pass

# ----------------------------
# 🔹 Enhanced SVM Classifiers with CORRECT PATHS
# ----------------------------

class SVMPlateTypeClassifier:
    """SVM for Plate Type Classification"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load trained SVM model and scaler from CORRECT paths"""
        try:
            # CORRECT PATHS for SVM Type Classification
            model_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\svm model\svm_model_balanced.pkl",
                r"C:\Users\DELL\Videos\systematic\thesis-project\SVM\svm_model_balanced.pkl",
            ]
            
            scaler_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\svm model\scaler.pkl",
                r"C:\Users\DELL\Videos\systematic\thesis-project\SVM\scaler.pkl",
            ]
            
            # Load model
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"✅ SVM Plate Type Classifier loaded from: {model_path}")
                    break
            else:
                print("❌ Could not find SVM model in any paths")
                self.model = None
            
            # Load scaler
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"✅ Scaler loaded from: {scaler_path}")
                    break
            else:
                print("❌ Scaler not found")
            
        except Exception as e:
            print(f"❌ Error loading SVM model: {e}")
            self.model = None
    
    def extract_features(self, image):
        """Extract HOG features from plate image"""
        try:
            if image is None or image.size == 0:
                return None
            
            # Resize to standard size
            resized = cv2.resize(image, (100, 50))
            
            # Convert to grayscale if needed
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            # Extract HOG features
            hog = cv2.HOGDescriptor((100, 50), (20, 20), (10, 10), (10, 10), 9)
            features = hog.compute(gray)
            
            return features.flatten() if features is not None else None
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def classify(self, image):
        """Classify plate type using SVM"""
        try:
            if image is None or image.size == 0:
                return {'classification': 'unknown', 'is_valid': False, 'confidence': 0.0}
            
            # Use SVM model if available
            if self.model is not None:
                features = self.extract_features(image)
                if features is not None and self.scaler is not None:
                    try:
                        # Scale features
                        features_scaled = self.scaler.transform([features])
                        
                        # Get prediction and probability
                        prediction = self.model.predict(features_scaled)[0]
                        probability = np.max(self.model.predict_proba(features_scaled))
                        
                        # Map prediction to class names
                        class_mapping = {
                            0: 'plate_government',
                            1: 'plate_motorcycle', 
                            2: 'plate_old',
                            3: 'plate_white',
                            4: 'plate_yellow'
                        }
                        
                        classification = class_mapping.get(prediction, 'plate_white')
                        is_valid = classification != 'plate_motorcycle'
                        
                        return {
                            'classification': classification,
                            'is_valid': is_valid,
                            'confidence': float(probability)
                        }
                    except Exception as e:
                        print(f"SVM prediction error: {e}")
            
            # FALLBACK: Use basic heuristics
            height, width = image.shape[:2]
            aspect_ratio = width / height if height > 0 else 0
            
            if aspect_ratio > 3.0:  # Very wide - likely motorcycle
                return {'classification': 'plate_motorcycle', 'is_valid': False, 'confidence': 0.7}
            elif aspect_ratio > 2.0:  # Standard vehicle plate
                return {'classification': 'plate_white', 'is_valid': True, 'confidence': 0.6}
            else:
                return {'classification': 'plate_white', 'is_valid': True, 'confidence': 0.5}
            
        except Exception as e:
            print(f"SVM classification error: {e}")
            return {'classification': 'plate_white', 'is_valid': True, 'confidence': 0.5}

class SVMCodingViolationDetector:
    """SVM for Coding Violation Detection"""
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load coding violation detection model from CORRECT paths"""
        try:
            # CORRECT PATHS for SVM Violation Detection
            model_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\SVM\svm_model_balanced.pkl",
                r"C:\Users\DELL\Videos\systematic\thesis-project\svm model\svm_model_balanced.pkl",
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"✅ SVM Coding Violation Detector loaded from: {model_path}")
                    break
            else:
                print("❌ Coding violation model not found")
                self.model = None
            
        except Exception as e:
            print(f"❌ Error loading coding violation model: {e}")
            self.model = None
    
    def detect_violation(self, plate_text, timestamp=None):
        """Detect coding violation using plate text"""
        try:
            if not plate_text:
                return {
                    'has_violation': False,
                    'violation_type': 'none',
                    'confidence': 0.0,
                    'reason': 'No plate text'
                }
            
            # Extract last digit for coding scheme
            last_char = plate_text[-1]
            if last_char.isdigit():
                last_digit = int(last_char)
            else:
                last_digit = 0
            
            # Get current time information
            current_time = datetime.now()
            hour = current_time.hour
            day_of_week = current_time.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
            
            # Philippine coding violation rules
            has_violation = False
            violation_type = 'none'
            confidence = 0.8
            
            # Common Philippine coding schemes
            if is_rush_hour:
                if last_digit in [1, 2]:  # Monday
                    if day_of_week == 0:
                        has_violation = True
                        violation_type = 'rush_hour_monday'
                elif last_digit in [3, 4]:  # Tuesday
                    if day_of_week == 1:
                        has_violation = True
                        violation_type = 'rush_hour_tuesday'
                elif last_digit in [5, 6]:  # Wednesday
                    if day_of_week == 2:
                        has_violation = True
                        violation_type = 'rush_hour_wednesday'
                elif last_digit in [7, 8]:  # Thursday
                    if day_of_week == 3:
                        has_violation = True
                        violation_type = 'rush_hour_thursday'
                elif last_digit in [9, 0]:  # Friday
                    if day_of_week == 4:
                        has_violation = True
                        violation_type = 'rush_hour_friday'
            
            return {
                'has_violation': has_violation,
                'violation_type': violation_type,
                'confidence': confidence,
                'last_digit': last_digit,
                'hour': hour,
                'is_rush_hour': bool(is_rush_hour),
                'is_weekend': bool(is_weekend),
                'day_of_week': day_of_week
            }
                
        except Exception as e:
            print(f"Coding violation detection error: {e}")
            return {
                'has_violation': False,
                'violation_type': 'none',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }

# ----------------------------
# 🔹 FIXED: Real-ESRGAN Enhancer with PROPER IMAGE HANDLING
# ----------------------------

class RealESRGANEnhancer:
    """Real-ESRGAN enhancer for license plate images - FIXED VERSION"""
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.load_model()
        self.enhancement_available = False
    
    def load_model(self):
        """Load Real-ESRGAN model - SIMPLIFIED VERSION"""
        try:
            if not TORCH_AVAILABLE:
                print("❌ PyTorch not available - Real-ESRGAN disabled")
                self.enhancement_available = False
                return
                
            # CORRECT PATHS for Real-ESRGAN
            model_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\real-cnn-model\RealESRGAN_x4plus.pth",
                r"C:\Users\DELL\Videos\systematic\thesis-project\real-cnn-model\net_g_100000.pth",
                r"C:\Users\DELL\Videos\systematic\thesis-project\real-cnn-model\model.pth",
            ]
            
            found_model = None
            for mp in model_paths:
                if os.path.exists(mp):
                    found_model = mp
                    print(f"✅ Found Real-ESRGAN model at: {mp}")
                    break
            
            if found_model:
                print(f"🔄 Loading Real-ESRGAN model from: {found_model}")
                
                # Simplified model creation - match common architectures
                try:
                    # Try to load using ultralytics implementation if available
                    try:
                        from basicsr.archs.rrdbnet_arch import RRDBNet
                        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                    except ImportError:
                        # Use our custom implementation
                        print("🔄 Using custom RRDBNet implementation")
                        self.model = SimpleRRDBNet(
                            num_in_ch=3, 
                            num_out_ch=3, 
                            num_feat=64, 
                            num_block=23,
                            num_grow_ch=32
                        )
                    
                    # Load weights
                    checkpoint = torch.load(found_model, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if 'params_ema' in checkpoint:
                        state_dict = checkpoint['params_ema']
                    elif 'params' in checkpoint:
                        state_dict = checkpoint['params']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    # Load with strict=False to ignore mismatches
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.eval()
                    self.model = self.model.to(self.device)
                    
                    self.enhancement_available = True
                    print(f"✅ Real-ESRGAN Model loaded successfully")
                    
                except Exception as e:
                    print(f"❌ Error loading Real-ESRGAN model: {e}")
                    print("🔄 Falling back to adaptive enhancement only")
                    self.model = None
                    self.enhancement_available = False
                
            else:
                print(f"❌ Real-ESRGAN model not found in any paths")
                print("🔄 Falling back to adaptive enhancement only")
                self.model = None
                self.enhancement_available = False
                
        except Exception as e:
            print(f"❌ Error loading Real-ESRGAN: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.enhancement_available = False
    
    def enhance_image(self, image):
        """Enhance image - FIXED with better error handling"""
        try:
            # Always apply adaptive enhancement first
            enhanced = self.adaptive_enhancement(image)
            
            # Try Real-ESRGAN if available
            if self.enhancement_available and self.model is not None:
                try:
                    esrgan_enhanced = self.real_esrgan_enhance(enhanced)
                    if esrgan_enhanced is not None:
                        print("  🎨 Real-ESRGAN Enhancement Applied")
                        return esrgan_enhanced
                    else:
                        print("  ⚠️ Real-ESRGAN failed, using adaptive enhancement")
                        return enhanced
                except Exception as e:
                    print(f"  ⚠️ Real-ESRGAN error: {e}, using adaptive enhancement")
                    return enhanced
            else:
                