import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image  # type: ignore
from datetime import datetime
import csv


from _07_training_model import MODEL_SAVE_DIR
from _05_1_data_preparation import get_data_generators

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5  # Classification threshold (50% probability)

SINGLE_IMAGE_PATH = r"C:\Users\Aditi Garg\atelectasis_detection\data\processed\before_balancing\Atelectasis\00002094_000.png"

PREDICTIONS_DIR = "data/predictions"
PREDICTION_IMAGES_DIR = os.path.join(PREDICTIONS_DIR, "images")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(PREDICTION_IMAGES_DIR, exist_ok=True)

PREDICTION_LOG_FILE = os.path.join(PREDICTIONS_DIR, "prediction_log.csv")


def get_latest_model_path(model_dir=MODEL_SAVE_DIR):
    """Finds the latest .keras model file in the specified directory."""
    keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError(f"No .keras model files found in {model_dir}")
    latest_file = max(keras_files, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
    latest_path = os.path.join(model_dir, latest_file)
    print(f"📁 Latest model found: {latest_file}")
    return latest_path


def preprocess_single_image(img_path):
    """Loads, resizes, and normalizes a single image for prediction."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Image file not found at: {img_path}")

    img = image.load_img(img_path, target_size=IMAGE_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array


def save_prediction_result(img_path, predicted_label, confidence):
    """Saves each prediction result to a CSV log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.exists(PREDICTION_LOG_FILE)
    with open(PREDICTION_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Image Name", "Predicted Label", "Confidence (%)", "Image Path"])
        writer.writerow([timestamp, os.path.basename(img_path), predicted_label, f"{confidence:.2f}", img_path])

    print(f"📝 Prediction logged in: {PREDICTION_LOG_FILE}")


def save_prediction_image(img_path, predicted_label, confidence):
    
    img_display = image.load_img(img_path, target_size=(250, 250))
    plt.figure(figsize=(5, 5))
    plt.imshow(img_display)
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)", fontsize=12, color="blue")
    plt.axis("off")

    output_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(img_path)}"
    output_path = os.path.join(PREDICTION_IMAGES_DIR, output_filename)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    print(f"🖼️ Saved prediction image to: {output_path}")


def predict_single_image(img_path):
    model_path = get_latest_model_path(MODEL_SAVE_DIR)
    print(f"\n🔹 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Get label order from data generators
    _, _, test_generator = get_data_generators()
    CLASS_LABELS = list(test_generator.class_indices.keys())
    print(f"Detected label order from data generator: {CLASS_LABELS}")

    # Preprocess image
    img_array = preprocess_single_image(img_path)

    # Predict
    print(f"\n📸 Predicting on image: {os.path.basename(img_path)}")
    raw_probability = model.predict(img_array)[0][0]
    confidence_pneumonia = raw_probability * 100

    # Determine label
    if raw_probability >= THRESHOLD:
        predicted_label = CLASS_LABELS[1]
        confidence = confidence_pneumonia
    else:
        predicted_label = CLASS_LABELS[0]
        confidence = 100.0 - confidence_pneumonia

    # Print results
    print("\n--- Prediction Result ---")
    print(f"🩻 Image: {os.path.basename(img_path)}")
    print(f"🔍 Predicted Label: {predicted_label}")
    print(f"📊 Model Confidence: {confidence:.2f}%")

    # ✅ Save to CSV and image folder
    save_prediction_result(img_path, predicted_label, confidence)
    save_prediction_image(img_path, predicted_label, confidence)

    return predicted_label, confidence


# ------------------------- MAIN EXECUTION -------------------------
if __name__ == "__main__":
    try:
        predict_single_image(SINGLE_IMAGE_PATH)
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
