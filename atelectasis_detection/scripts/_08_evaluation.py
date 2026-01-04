import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns # type: ignore
import os
from datetime import datetime

# Import from your existing modules
from _05_1_data_preparation import get_data_generators
from _07_training_model import MODEL_SAVE_DIR

# Constants
_, _, test_generator = get_data_generators()
CLASS_LABELS = list(test_generator.class_indices.keys())
print("Detected label order:", CLASS_LABELS)

def get_latest_model_path(model_dir=MODEL_SAVE_DIR):
    """
    Finds the latest .keras model file in the specified directory.
    """
    keras_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    if not keras_files:
        raise FileNotFoundError(f"No .keras model files found in {model_dir}")
    latest_file = max(keras_files, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
    return os.path.join(model_dir, latest_file)

def evaluate_atelectasis_model():
    """
    Loads the best trained model, evaluates it on the test set, 
    and reports accuracy, precision, recall, confusion matrix, and ROC-AUC.
    """
    print("🔹 Loading test data generator...")
    _, _, test_generator = get_data_generators()

    test_steps = test_generator.samples // test_generator.batch_size

    # Load latest model automatically
    model_path = get_latest_model_path(MODEL_SAVE_DIR)
    print(f"\n📁 Loading best model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Evaluate
    print("\n🔹 Evaluating model on test set...")
    results = model.evaluate(test_generator, steps=test_steps, verbose=1)
    metrics_names = model.metrics_names
    print("\n--- Core Evaluation Metrics ---")
    for name, value in zip(metrics_names, results):
        print(f"{name.capitalize()}: {value:.4f}")

    # Reset generator for fresh predictions
    test_generator.reset()

    # Raw prediction probabilities (for ROC)
    Y_pred_proba = model.predict(test_generator, steps=test_steps + 1)
    Y_pred_classes = (Y_pred_proba > 0.5).astype(int).flatten()
    Y_true = test_generator.classes[:len(Y_pred_classes)]

    # Classification report
    print("\n--- Detailed Classification Report ---")
    print(classification_report(Y_true, Y_pred_classes, target_names=CLASS_LABELS, digits=4))

    # Confusion matrix
    cm = confusion_matrix(Y_true, Y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save confusion matrix
    cm_path = os.path.join(MODEL_SAVE_DIR, f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"📊 Confusion matrix saved to: {cm_path}")

    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(Y_true, Y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc="lower right")

    # Save ROC curve
    roc_path = os.path.join(MODEL_SAVE_DIR, f'roc_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(roc_path, dpi=300)
    plt.show()
    print(f"📈 ROC curve saved to: {roc_path}")

    print(f"\n✅ Area Under the Curve (AUC): {roc_auc:.4f}")
    print("\n🎯 Evaluation completed successfully.")

if __name__ == '__main__':
    evaluate_atelectasis_model() 
    
