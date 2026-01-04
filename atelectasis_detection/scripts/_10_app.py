import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image

# Import your prediction function
from _09_prediction_atelectasis import predict_single_image

# ------------------ Config ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data/uploads")
PREDICTIONS_FOLDER = os.path.join(BASE_DIR, "data/predictions")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ Helpers ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_overlay_image(input_path, predicted_label, confidence):
    """Reads uploaded image, overlays label and confidence, saves to predictions folder."""
    img = Image.open(input_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{predicted_label} ({confidence:.2f}%)", fontsize=16, color="red")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_{timestamp}.png"
    save_path = os.path.join(PREDICTIONS_FOLDER, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return filename

# ------------------ Routes ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "xray-file-input" not in request.files:
        flash("No file part in the request")
        return redirect(request.url)

    file = request.files["xray-file-input"]

    if file.filename == "":
        flash("No file selected")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # ---- Run prediction ----
        predicted_label, confidence = predict_single_image(file_path)

        # ---- Create and save overlay image ----
        overlay_filename = save_overlay_image(file_path, predicted_label, confidence)
        overlay_url = url_for("uploaded_prediction", filename=overlay_filename)

        return render_template(
            "index.html",
            prediction=True,
            predicted_label=predicted_label,
            confidence=confidence,
            overlay_url=overlay_url
        )

    flash("Allowed image types are png, jpg, jpeg")
    return redirect(request.url)

@app.route("/uploads/<filename>")
def uploaded_input(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/predictions/<filename>")
def uploaded_prediction(filename):
    return send_from_directory(PREDICTIONS_FOLDER, filename)

# ------------------ Main ------------------
if __name__ == "__main__":
    app.run(debug=True)
