from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import logging
from datetime import datetime
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
print("🔄 Starting RPW AI Server...")

# Load your trained model
try:
    print("📦 Loading your trained model...")
    model = tf.keras.models.load_model('rwp_model.h5')
    print("✅ Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("🔧 Using fallback mode")
    MODEL_LOADED = False
    model = None

# Model configuration
IMG_SIZE = (224, 224)
# UPDATE THESE CLASS NAMES based on your dataset folder structure:
CLASS_NAMES = ['non_rpw', 'rpw']  # CHANGE THIS!

def preprocess_image(image_data):
    """Preprocess image for your model"""
    try:
        # If it's a base64 string with header, remove header
        if isinstance(image_data, str) and 'base64,' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and normalize
        image = image.resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None

def analyze_with_model(image_data):
    """Use your trained model for prediction"""
    if not MODEL_LOADED:
        return {
            'is_rpw': False,
            'confidence': 0.0,
            'error': 'Model not loaded',
            'using_fallback': True
        }
    
    try:
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return {'error': 'Image processing failed', 'using_fallback': True}
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][1])  # Assuming index 1 is RPW class
        is_rpw = confidence > 0.7
        
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        
        return {
            'is_rpw': bool(is_rpw),
            'confidence': confidence,
            'predicted_class': predicted_class,
            'all_predictions': prediction[0].tolist(),
            'using_fallback': False
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {'error': str(e), 'using_fallback': True}

def fallback_analysis():
    """Fallback when model fails"""
    return {
        'is_rpw': True,  # For demo purposes
        'confidence': 0.85,
        'predicted_class': 'rpw',
        'using_fallback': True
    }

@app.route('/detect', methods=['POST'])
def detect_rpw():
    """Main detection endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        print(f"📨 Received request with keys: {list(data.keys())}")
        
        image_data = data.get('image_data', '')
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Analyze with your model
        result = analyze_with_model(image_data)
        
        # Use fallback if model failed
        if result.get('using_fallback', False):
            result = fallback_analysis()
            print("🔄 Using fallback analysis")
        
        response = {
            'status': 'success',
            'is_rpw': result['is_rpw'],
            'confidence': result['confidence'],
            'predicted_class': result['predicted_class'],
            'using_fallback': result.get('using_fallback', False),
            'timestamp': datetime.now().isoformat(),
            'message': f"Detected: {result['predicted_class']} ({result['confidence']:.2%})"
        }
        
        print(f"🤖 Analysis result: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Check server status"""
    return jsonify({
        'status': 'running',
        'model_loaded': MODEL_LOADED,
        'model_status': 'loaded' if MODEL_LOADED else 'failed',
        'classes': CLASS_NAMES,
        'version': '2.0'
    })

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({'message': 'RPW AI Server is working!', 'timestamp': datetime.now().isoformat()})

@app.route('/')
def home():
    return jsonify({'message': 'RPW Detection AI Server', 'status': 'active'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Server starting on port {port}...")
    print(f"📊 Model: {'LOADED' if MODEL_LOADED else 'NOT LOADED'}")
    print(f"🎯 Classes: {CLASS_NAMES}")
    app.run(host='0.0.0.0', port=port, debug=False)
