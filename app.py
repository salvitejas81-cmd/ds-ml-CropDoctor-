import os
from flask import Flask, render_template, request, url_for, redirect, session, flash
from werkzeug.utils import secure_filename

from prediction import predict_disease
from treatment import get_treatment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = "cropdoctor_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "Dataset", "PlantVillage", "train")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRAPH_FOLDER = os.path.join(BASE_DIR, "static", "graphs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "admin123":
            session["logged_in"] = True
            return redirect(url_for("home"))

        flash("Invalid username or password")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("index.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if "image" not in request.files:
        return render_template("index.html", error="Please select an image.")

    uploaded_image = request.files["image"]

    if uploaded_image.filename == "":
        return render_template("index.html", error="Please select an image.")

    filename = secure_filename(uploaded_image.filename)

    if filename == "":
        return render_template("index.html", error="Invalid image filename.")

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_image.save(image_path)

    try:
        disease, confidence = predict_disease(image_path)
        treatment_data = get_treatment(disease)

        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "index.html",
            prediction=disease,
            confidence=confidence,
            symptoms=treatment_data.get("symptoms", "Symptoms information is currently unavailable."),
            treatment=treatment_data.get("treatment", "Treatment information is currently unavailable."),
            prevention=treatment_data.get("prevention", "Prevention information is currently unavailable."),
            image_path=image_url
        )

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


def create_dashboard_graphs():
    class_names = []
    image_counts = []

    if not os.path.exists(TRAIN_PATH):
        print("Dataset folder not found:", TRAIN_PATH)
        return

    for class_name in sorted(os.listdir(TRAIN_PATH)):
        class_path = os.path.join(TRAIN_PATH, class_name)

        if not os.path.isdir(class_path):
            continue

        count = 0

        for filename in os.listdir(class_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1

        class_names.append(class_name)
        image_counts.append(count)

    if not class_names:
        print("No dataset images found.")
        return

    display_names = [
        name.replace("___", " - ").replace("_", " ")
        for name in class_names
    ]

    plt.figure(figsize=(16, 8))
    plt.bar(display_names, image_counts)
    plt.xlabel("Disease Classes")
    plt.ylabel("Number of Images")
    plt.title("Disease-wise Image Distribution")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "disease_distribution.png"), dpi=150)
    plt.close()

    healthy_images = 0
    diseased_images = 0

    for name, count in zip(class_names, image_counts):
        if "healthy" in name.lower():
            healthy_images += count
        else:
            diseased_images += count

    plt.figure(figsize=(7, 7))
    plt.pie(
        [healthy_images, diseased_images],
        labels=["Healthy", "Diseased"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Healthy vs Diseased Images")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "healthy_diseased.png"), dpi=150)
    plt.close()

    data = list(zip(class_names, image_counts))
    data.sort(key=lambda item: item[1], reverse=True)

    top10 = data[:10]

    top_names = [
        name.replace("___", " - ").replace("_", " ")
        for name, count in top10
    ]

    top_counts = [
        count
        for name, count in top10
    ]

    plt.figure(figsize=(12, 7))
    plt.bar(top_names, top_counts)
    plt.xlabel("Disease Classes")
    plt.ylabel("Number of Images")
    plt.title("Top 10 Disease Classes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "top10_classes.png"), dpi=150)
    plt.close()

    print("Dashboard graphs created successfully!")


@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    class_counts = {}

    if not os.path.exists(TRAIN_PATH):
        return "Dataset folder not found: " + TRAIN_PATH

    for class_name in sorted(os.listdir(TRAIN_PATH)):
        class_path = os.path.join(TRAIN_PATH, class_name)

        if not os.path.isdir(class_path):
            continue

        count = 0

        for filename in os.listdir(class_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1

        class_counts[class_name] = count

    total_images = sum(class_counts.values())
    total_diseases = len(class_counts)

    healthy_images = sum(
        count
        for disease, count in class_counts.items()
        if "healthy" in disease.lower()
    )

    diseased_images = total_images - healthy_images

    sorted_diseases = sorted(
        class_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    create_dashboard_graphs()

    return render_template(
        "dashboard.html",
        total_images=total_images,
        total_diseases=total_diseases,
        healthy_images=healthy_images,
        diseased_images=diseased_images,
        sorted_diseases=sorted_diseases
    )


if __name__ == "__main__":
    app.run(debug=True)