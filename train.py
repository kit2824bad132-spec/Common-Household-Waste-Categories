"""
train.py
=============================================================================
Household Waste Classification using MobileNetV2 Transfer Learning
=============================================================================
This script:
1. Discovers the dataset categories in dataset/archive/garbage_classification
2. Splits dataset into Train (70%), Validation (15%), and Test (15%) with stratification
3. Prevents data leakage by holding out the test set completely
4. Applies data augmentation and MobileNetV2 preprocessing
5. Trains a lightweight transfer learning model using MobileNetV2
6. Evaluates actual metrics (Accuracy, Precision, Recall, F1, Confusion Matrix) on the test set
7. Saves the trained model to 'waste_classifier.keras' and saves 'class_names.json'
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Plotting without GUI window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Dataset paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'archive', 'garbage_classification')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'waste_classifier.keras')
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, 'class_names.json')
METRICS_SAVE_PATH = os.path.join(PROJECT_ROOT, 'evaluation_results.json')
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, 'static', 'confusion_matrix.png')
HISTORY_PLOT_PATH = os.path.join(PROJECT_ROOT, 'static', 'training_history.png')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'test_split.json')

# Recycling & bin guidance dictionary for all 12 waste categories
WASTE_STREAM_MAPPING = {
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


def discover_dataset(dataset_dir):
    """
    Inspect the dataset folder and discover class directories and valid image files.
    Returns: file_paths, labels, class_names
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not classes:
        raise ValueError(f"No category folders found in {dataset_dir}")

    file_paths = []
    labels = []
    class_counts = {}

    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_dir, class_name)
        images = [
            os.path.join(class_folder, f) for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        class_counts[class_name] = len(images)
        for img_path in images:
            file_paths.append(img_path)
            labels.append(idx)

    print(f"\n=======================================================")
    print(f"Discovered {len(classes)} Waste Categories:")
    print(f"=======================================================")
    for c in classes:
        print(f"  • {c:15s}: {class_counts[c]} images")
    print(f"Total dataset size: {len(file_paths)} images\n")

    return np.array(file_paths), np.array(labels), classes


def create_splits(file_paths, labels, class_names, test_size=0.15, val_size=0.15, max_samples_per_class=None):
    """
    Create stratified train, val, test splits ensuring NO data leakage.
    Optionally balance/cap samples per class for rapid transfer learning if requested.
    """
    from sklearn.model_selection import train_test_split

    # If max_samples_per_class is specified, subsample evenly across classes
    if max_samples_per_class:
        sub_paths = []
        sub_labels = []
        for idx in range(len(class_names)):
            idx_mask = (labels == idx)
            idx_paths = file_paths[idx_mask]
            if len(idx_paths) > max_samples_per_class:
                chosen = np.random.RandomState(42).choice(idx_paths, size=max_samples_per_class, replace=False)
            else:
                chosen = idx_paths
            sub_paths.extend(chosen)
            sub_labels.extend([idx] * len(chosen))
        file_paths = np.array(sub_paths)
        labels = np.array(sub_labels)
        print(f"Balanced/Subsampled to {len(file_paths)} total images (max {max_samples_per_class} per class).")

    # Step 1: Split into (Train + Val) and Test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # Step 2: Split (Train + Val) into Train and Val
    adjusted_val_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=adjusted_val_size, stratify=train_val_labels, random_state=42
    )

    print(f"Data Splits:")
    print(f"  • Train set      : {len(train_paths)} samples ({len(train_paths)/len(file_paths)*100:.1f}%)")
    print(f"  • Validation set : {len(val_paths)} samples ({len(val_paths)/len(file_paths)*100:.1f}%)")
    print(f"  • Test set       : {len(test_paths)} samples ({len(test_paths)/len(file_paths)*100:.1f}%)")

    # Save test set filepaths and labels for independent reproducible evaluation
    test_record = {
        'paths': [os.path.relpath(p, PROJECT_ROOT) for p in test_paths],
        'labels': [int(l) for l in test_labels],
        'class_names': class_names
    }
    with open(TEST_DATA_PATH, 'w') as f:
        json.dump(test_record, f, indent=2)
    print(f"Saved test set split to {TEST_DATA_PATH} (ensures zero data leakage).")

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def build_tf_dataset(file_paths, labels, batch_size=32, is_training=False, img_size=(224, 224)):
    """
    Construct high-performance tf.data pipeline with MobileNetV2 preprocessing.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    def load_and_preprocess(path, label):
        raw_img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw_img, channels=3)
        img = tf.image.resize(img, img_size)
        img = preprocess_input(img)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        # Random brightness and slight contrast adjustment
        img = tf.image.random_brightness(img, 0.1)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 2000), seed=42)
    
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_mobilenetv2_model(num_classes, input_shape=(224, 224, 3)):
    """
    Build transfer learning model with MobileNetV2 backbone.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    print("\nInitializing MobileNetV2 base model (pre-trained on ImageNet)...")
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base feature extractor
    base_model.trainable = False

    # Build custom classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='MobileNetV2_WasteClassifier')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def plot_and_save_history(history, save_path):
    """Plot and save training and validation accuracy & loss curves."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, label='Training Accuracy', color='#059669', linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', color='#2563eb', linewidth=2, linestyle='--')
    ax1.set_title('Model Accuracy per Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Loss Plot
    ax2.plot(epochs_range, loss, label='Training Loss', color='#dc2626', linewidth=2)
    ax2.plot(epochs_range, val_loss, label='Validation Loss', color='#d97706', linewidth=2, linestyle='--')
    ax2.set_title('Model Loss per Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def evaluate_and_report(model, test_dataset, test_paths, test_labels, class_names):
    """
    Evaluate actual test performance on the test set.
    Generates precision, recall, f1-score, and confusion matrix.
    Never invents results.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n=======================================================")
    print("Evaluating Model on Independent Test Set...")
    print("=======================================================")

    # Get predictions
    raw_preds = model.predict(test_dataset)
    y_pred = np.argmax(raw_preds, axis=1)
    y_true = np.array(test_labels)

    # Classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    print("\nDetailed Test Classification Report:")
    print(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix on Isolated Test Set',
        ylabel='True Waste Category',
        xlabel='Predicted Waste Category'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text numbers into matrix cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    # Save evaluation summary to JSON
    overall_accuracy = float(report_dict['accuracy'])
    macro_f1 = float(report_dict['macro avg']['f1-score'])
    weighted_f1 = float(report_dict['weighted avg']['f1-score'])

    eval_data = {
        'test_samples': len(y_true),
        'accuracy': overall_accuracy,
        'macro_precision': float(report_dict['macro avg']['precision']),
        'macro_recall': float(report_dict['macro avg']['recall']),
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': {
            c: {
                'precision': float(report_dict[c]['precision']),
                'recall': float(report_dict[c]['recall']),
                'f1_score': float(report_dict[c]['f1-score']),
                'support': int(report_dict[c]['support'])
            } for c in class_names
        },
        'confusion_matrix': cm.tolist()
    }

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(eval_data, f, indent=2)
    print(f"Saved evaluation metrics to {METRICS_SAVE_PATH}")

    return eval_data


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on Household Waste Dataset")
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_DIR, help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default 5)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default 32)')
    parser.add_argument('--max-samples-per-class', type=int, default=None, 
                        help='Optional cap per class (e.g. 250 for fast CPU training, or None for full dataset)')
    parser.add_argument('--fine-tune-epochs', type=int, default=0, help='Optional unfreezing fine-tuning epochs')
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras import callbacks

    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) Detected: {gpus}")
    else:
        print("Running on CPU.")

    # 1. Discover dataset
    file_paths, labels, class_names = discover_dataset(args.dataset)

    # Save class names JSON immediately
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to {CLASS_NAMES_PATH}")

    # Save full waste metadata for app.py
    metadata_path = os.path.join(PROJECT_ROOT, 'waste_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(WASTE_STREAM_MAPPING, f, indent=2)

    # 2. Split dataset with stratification
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_splits(
        file_paths, labels, class_names, max_samples_per_class=args.max_samples_per_class
    )

    # 3. Create tf.data datasets
    train_ds = build_tf_dataset(train_paths, train_labels, batch_size=args.batch_size, is_training=True)
    val_ds = build_tf_dataset(val_paths, val_labels, batch_size=args.batch_size, is_training=False)
    test_ds = build_tf_dataset(test_paths, test_labels, batch_size=args.batch_size, is_training=False)

    # 4. Build MobileNetV2 transfer learning model
    model, base_model = build_mobilenetv2_model(num_classes=len(class_names))
    model.summary()

    # 5. Training Callbacks
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # 6. Train Head (Transfer Learning)
    print(f"\nStarting MobileNetV2 Training ({args.epochs} epochs)...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb_list
    )
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes).")

    # 7. Optional Fine-Tuning (if requested)
    if args.fine_tune_epochs > 0:
        print(f"\nFine-tuning top 30 layers of MobileNetV2 for {args.fine_tune_epochs} epochs...")
        base_model.trainable = True
        # Freeze all layers except the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=cb_list
        )
        # Append histories
        for k in history.history:
            history.history[k].extend(ft_history.history[k])

    # 8. Save model explicitly in native Keras format
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved successfully as: {MODEL_SAVE_PATH}")

    # 9. Plot training history
    plot_and_save_history(history, HISTORY_PLOT_PATH)

    # 10. Evaluate on the separate test set
    eval_results = evaluate_and_report(model, test_ds, test_paths, test_labels, class_names)

    print("\n=======================================================")
    print("FINAL TEST EVALUATION SUMMARY:")
    print(f"  • Test Accuracy  : {eval_results['accuracy']*100:.2f}%")
    print(f"  • Macro F1-Score : {eval_results['macro_f1']*100:.2f}%")
    print(f"  • Weighted F1    : {eval_results['weighted_f1']*100:.2f}%")
    print("=======================================================")


if __name__ == '__main__':
    main()
