import os
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Basic paths and configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'uploads')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'waste_classifier.keras')
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, 'class_names.json')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'waste_metadata.json')
EVAL_RESULTS_PATH = os.path.join(PROJECT_ROOT, 'evaluation_results.json')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Max 16MB upload

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'household-waste-classifier-eco-2026'

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and metadata caching
_MODEL = None
_CLASS_NAMES = None
_METADATA = None


# Default fallback recycling guidance dictionary for all 12 dataset categories
DEFAULT_METADATA = {
    'battery': {
        'bin_color': 'Red',
        'stream': 'Hazardous & E-Waste',
        'badge_class': 'hazardous',
        'guidance': 'Never dispose of batteries in general household bins! Take to an authorized e-waste drop-off box or battery collection depot to prevent toxic lithium/acid leakage.'
    },
    'biological': {
        'bin_color': 'Green',
        'stream': 'Organic / Compostable',
        'badge_class': 'organic',
        'guidance': 'Place in the green organic/wet bin. Ideal for home composting or municipal biomethanation processing.'
    },
    'brown-glass': {
        'bin_color': 'Blue',
        'stream': 'Recyclable Glass',
        'badge_class': 'recyclable',
        'guidance': 'Rinse thoroughly, remove metal caps, and place in the dry recyclables blue bin for color-separated glass recycling.'
    },
    'cardboard': {
        'bin_color': 'Blue',
        'stream': 'Paper & Cardboard',
        'badge_class': 'recyclable',
        'guidance': 'Flatten boxes to save space. Keep dry and free of food oil or grease before depositing in the recyclables bin.'
    },
    'clothes': {
        'bin_color': 'Yellow/Purple',
        'stream': 'Textile Reuse / Recycling',
        'badge_class': 'textile',
        'guidance': 'If in wearable condition, donate to local charities or shelters. Torn garments can be deposited in municipal textile recycling banks.'
    },
    'green-glass': {
        'bin_color': 'Blue',
        'stream': 'Recyclable Glass',
        'badge_class': 'recyclable',
        'guidance': 'Rinse clean and discard in designated glass recycling stream. 100% infinitely recyclable.'
    },
    'metal': {
        'bin_color': 'Blue',
        'stream': 'Dry Recyclables (Metals)',
        'badge_class': 'recyclable',
        'guidance': 'Empty and rinse aluminum cans or food tins. Crush cans if possible to optimize bin volume.'
    },
    'paper': {
        'bin_color': 'Blue',
        'stream': 'Dry Recyclables (Paper)',
        'badge_class': 'recyclable',
        'guidance': 'Keep clean and dry. Avoid recycling paper contaminated with food residue, wax, or chemical adhesives.'
    },
    'plastic': {
        'bin_color': 'Blue',
        'stream': 'Dry Recyclables (Plastics)',
        'badge_class': 'recyclable',
        'guidance': 'Rinse bottles and containers. Check recycling triangle (#1 PET, #2 HDPE are widely recyclable). Replace caps securely.'
    },
    'shoes': {
        'bin_color': 'Yellow/Purple',
        'stream': 'Footwear & Textiles',
        'badge_class': 'textile',
        'guidance': 'Tie pairs together by laces. Donate usable pairs or take to shoe recycling programs that shred soles into playground turf.'
    },
    'trash': {
        'bin_color': 'Black/Grey',
        'stream': 'Residual / Non-Recyclable',
        'badge_class': 'landfill',
        'guidance': 'Non-recyclable household waste destined for sanitary landfill or waste-to-energy incineration. Segregate strictly to avoid contaminating recyclables.'
    },
    'white-glass': {
        'bin_color': 'Blue',
        'stream': 'Recyclable Glass (Clear)',
        'badge_class': 'recyclable',
        'guidance': 'Rinse clean. Clear flint glass is highly valuable for remelting into new food containers and glass jars.'
    }
}


def load_classifier_resources():
    """Lazily load the trained Keras model, class labels, and domain metadata."""
    global _MODEL, _CLASS_NAMES, _METADATA

    # 1. Load Class Names
    if _CLASS_NAMES is None and os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, 'r') as f:
                _CLASS_NAMES = json.load(f)
        except Exception as e:
            print(f"Warning loading class_names.json: {e}")

    # 2. Load Metadata
    if _METADATA is None:
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, 'r') as f:
                    _METADATA = json.load(f)
            except Exception:
                _METADATA = DEFAULT_METADATA
        else:
            _METADATA = DEFAULT_METADATA

    # 3. Load Trained Keras Model
    if _MODEL is None and os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            print(f"Loading trained model from {MODEL_PATH}...")
            _MODEL = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            _MODEL = None

    return _MODEL, _CLASS_NAMES, _METADATA


def allowed_file(filename):
    """Verify if the uploaded file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image_for_mobilenet(filepath):
    """
    Load an image, convert to RGB, resize to 224x224, and apply
    MobileNetV2 normalized preprocessing ([-1, 1] scaling).
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Open image using Pillow to handle arbitrary dimensions & orientations safely
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    
    img_array = np.array(img, dtype=np.float32)
    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    # MobileNetV2 preprocessing: (x / 127.5) - 1.0
    preprocessed_batch = preprocess_input(img_batch)
    return preprocessed_batch


@app.route('/')
def index():
    """Render the main home page."""
    return render_template('index.html')


@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check current model status, test evaluation metrics, and available classes."""
    model, class_names, _ = load_classifier_resources()
    eval_metrics = None

    if os.path.exists(EVAL_RESULTS_PATH):
        try:
            with open(EVAL_RESULTS_PATH, 'r') as f:
                eval_metrics = json.load(f)
        except Exception:
            pass

    return jsonify({
        'model_loaded': model is not None,
        'model_path_exists': os.path.exists(MODEL_PATH),
        'class_count': len(class_names) if class_names else 0,
        'class_names': class_names or [],
        'evaluation': eval_metrics
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for classifying household waste images using MobileNetV2.
    """
    if 'waste_image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file found in the request.'
        }), 400

    file = request.files['waste_image']

    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file was selected.'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Unsupported file format. Allowed formats: {", ".join(ALLOWED_EXTENSIONS).upper()}'
        }), 400

    try:
        # Save file securely for preview and inference
        filename = secure_filename(file.filename)
        unique_filename = f"upload_{int(np.random.randint(1000, 9999))}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        image_url = f"/static/uploads/{unique_filename}"

        # Load model and class names
        model, class_names, metadata = load_classifier_resources()

        # If model is not yet trained or saved
        if model is None or not class_names:
            return jsonify({
                'success': True,
                'model_connected': False,
                'status': 'Model Training / Pending',
                'filename': filename,
                'image_url': image_url,
                'message': (
                    'Image received successfully! However, the trained model (waste_classifier.keras) '
                    'is not connected yet. Run "python train.py" to train and save the model.'
                ),
                'placeholder_data': {
                    'demo_category': 'Model Not Found',
                    'demo_bin': 'Pending',
                    'demo_confidence': 0.0,
                    'guide': 'Train the model by executing: python train.py'
                }
            })

        # Model IS connected: Perform real inference
        preprocessed = preprocess_image_for_mobilenet(filepath)
        predictions = model.predict(preprocessed)
        probabilities = predictions[0]

        top_idx = int(np.argmax(probabilities))
        top_category = class_names[top_idx]
        confidence_score = float(probabilities[top_idx])
        confidence_percent = round(confidence_score * 100, 1)

        # Get top-3 predictions for distribution analysis
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_3_predictions = [
            {
                'category': class_names[i],
                'display_name': class_names[i].replace('-', ' ').title(),
                'confidence': round(float(probabilities[i]) * 100, 1)
            }
            for i in top_3_indices
        ]

        # Retrieve domain recycling advice
        cat_meta = metadata.get(top_category, {
            'bin_color': 'Blue / Green',
            'stream': 'General Waste',
            'badge_class': 'recyclable',
            'guidance': 'Inspect local municipal guidelines for appropriate disposal.'
        })

        return jsonify({
            'success': True,
            'model_connected': True,
            'status': 'Inference Successful',
            'filename': filename,
            'image_url': image_url,
            'prediction': {
                'category': top_category,
                'display_name': top_category.replace('-', ' ').title(),
                'confidence': confidence_percent,
                'bin_color': cat_meta.get('bin_color', 'Blue'),
                'stream': cat_meta.get('stream', 'Recyclable'),
                'badge_class': cat_meta.get('badge_class', 'recyclable'),
                'guidance': cat_meta.get('guidance', ''),
                'top_3': top_3_predictions
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred during inference: {str(e)}'
        }), 500


@app.route('/api/contact', methods=['POST'])
def contact():
    """Handle contact form submissions."""
    data = request.get_json() or request.form

    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    subject = data.get('subject', '').strip()
    message = data.get('message', '').strip()

    if not name or not email or not message:
        return jsonify({
            'success': False,
            'error': 'Please fill in all required fields (Name, Email, Message).'
        }), 400

    return jsonify({
        'success': True,
        'message': f'Thank you, {name}! Your message has been received. We will get back to you soon.'
    })


if __name__ == '__main__':
    # Preload model if available
    load_classifier_resources()
    app.run(debug=True, host='127.0.0.1', port=5000)
