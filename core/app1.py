import os
import csv
import boto3
import json
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file
from celery import Celery
import psycopg2
from io import BytesIO  # Import BytesIO for handling binary data
import base64  # Import base64 for encoding binary data
import ollama
from PIL import Image  # Import Pillow for image processing
from google.cloud import vision  # Import Google Cloud Vision
import dashscope  # Import DashScope for Qwen API

# Flask Configurations
UPLOAD_FOLDER = os.path.abspath("static/uploads/")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app = Flask(__name__)
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Max file size 16MB

# Celery Configuration
app.config["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
app.config["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"


# Initialize Celery
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config["CELERY_RESULT_BACKEND"],
        broker=app.config["CELERY_BROKER_URL"],
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


celery = make_celery(app)

# PostgreSQL Database Connection Details
DB_HOST = "localhost"
DB_NAME = "pixcy"  # Replace with your database name
DB_USER = "postgres"  # Replace with your database username
DB_PASSWORD = "Sathwik@20022004"  # Replace with your database password


# Load AWS Credentials from CSV
def load_aws_credentials(filename="aws_credentials.csv"):
    try:
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            aws_access_key, aws_secret_key = next(reader)
            return aws_access_key, aws_secret_key
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please check the file path.")
        return None, None
    except Exception as e:
        print(f"Error loading AWS credentials: {e}")
        return None, None


AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = load_aws_credentials(
    r"C:\Users\sathw\OneDrive\Desktop\Tessrac Internship\Django1\aws_credentials.csv"
)
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    rekognition = boto3.client(
        "rekognition",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )
else:
    raise ValueError("AWS credentials not found! Check 'aws_credentials.csv'.")


vision_client = vision.ImageAnnotatorClient()

# Set up Qwen API
dashscope.api_key = "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"  # Replace with your Qwen API key


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def fetch_all_images_from_postgres():
    """Fetch all images' paths, binary data, and predefined tags from PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("SELECT image_path, image_blob, tags FROM images")
        images = cursor.fetchall()
        cursor.close()
        conn.close()
        return images
    except Exception as e:
        print(f"Error fetching images from PostgreSQL: {e}")
        return []


def update_image_in_postgres(
    image_path, labels_with_confidence, meta_title, meta_description
):
    """Update an image's metadata in PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()
        # Check if the 'confidence_scores' column exists
        cursor.execute(
            """
            SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'images' AND column_name = 'confidence_scores'
            );
            """
        )
        confidence_scores_exists = cursor.fetchone()[0]
        # Construct the update query dynamically based on column existence
        if confidence_scores_exists:
            update_query = """
            UPDATE images
            SET labels = %s, "Meta-Title" = %s, "Meta-Description" = %s, confidence_scores = %s
            WHERE image_path = %s;
            """
            cursor.execute(
                update_query,
                (
                    json.dumps(
                        [label for label, _ in labels_with_confidence]
                    ),  # Store labels
                    meta_title,
                    meta_description,
                    json.dumps(
                        {label: score for label, score in labels_with_confidence}
                    ),  # Confidence scores
                    image_path,
                ),
            )
        else:
            update_query = """
            UPDATE images
            SET labels = %s, "Meta-Title" = %s, "Meta-Description" = %s
            WHERE image_path = %s;
            """
            cursor.execute(
                update_query,
                (
                    json.dumps(
                        [label for label, _ in labels_with_confidence]
                    ),  # Store labels
                    meta_title,
                    meta_description,
                    image_path,
                ),
            )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error updating image in PostgreSQL: {e}")


def generate_meta_tags(labels_with_confidence):
    """
    Generate SEO-optimized meta title and description using Ollama API.
    """
    if not labels_with_confidence:
        return "Unknown Image", "No recognizable objects detected."

    # Filter labels based on confidence score ranges
    filtered_labels = []
    for label, confidence in labels_with_confidence:
        if confidence >= 90:
            filtered_labels.append(label)
        elif 75 <= confidence < 90:
            # Include only if it supports other high-confidence labels
            if any(conf >= 90 for _, conf in labels_with_confidence):
                filtered_labels.append(label)
        elif 55 <= confidence < 75:
            # Use only if highly contextually relevant
            if "fashion" in label.lower() or "clothing" in label.lower():
                filtered_labels.append(label)

    prompt = f"""
    You are an SEO expert. Generate an SEO-optimized meta title and short description
    for an image containing: {', '.join(filtered_labels)}.
    Follow these rules:
    - The title should be concise, engaging, and include relevant keywords.
    - The description should be a brief summary of the image content, including keywords naturally.
    Provide output in this format:
    Meta Title: 
    Meta Description: 
    """
    try:
        # Call Ollama API to generate meta tags
        response = ollama.chat(
            model="mistral",  # Replace with your preferred model
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response["message"]["content"].strip()
        # Parse the response to extract meta title and description
        meta_title, meta_description = "Unknown Image", "No description available."
        for line in response_text.split("\n"):
            if "Meta Title:" in line:
                meta_title = line.replace("Meta Title:", "").strip() or "Unknown Image"
            elif "Meta Description:" in line:
                meta_description = (
                    line.replace("Meta Description:", "").strip()
                    or "No description available."
                )
        return meta_title, meta_description
    except Exception as e:
        error_message = f"Error generating meta tags: {e}"
        print(error_message)  # Log to console
        return "Unknown Image", "No description available."


def generate_labels_with_qwen(image_blob):
    """
    Generate labels for an image using Qwen.
    """
    try:
        # Convert image to Base64
        encoded_image = base64.b64encode(image_blob).decode("utf-8")
        # Call Qwen API using MultiModalConversation
        response = dashscope.MultiModalConversation.call(
            api_key=dashscope.api_key,
            model="qwen-vl",  # Specify the model explicitly
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": encoded_image},
                        {"text": "Describe the contents of this image."},
                    ],
                }
            ],
        )
        # Extract labels from the response
        labels = []
        if response.status_code == 200:
            for item in response.output.choices:
                labels.append(item["message"]["content"])
        return [
            {"Name": label, "Confidence": 0.0} for label in labels
        ]  # Default confidence 0.0
    except Exception as e:
        print(f"Qwen label generation failed: {str(e)}")
        return []


@app.route("/process_all_images", methods=["GET"])
def process_all_images():
    try:
        # Fetch all images from the database
        images = fetch_all_images_from_postgres()
        if not images:
            return "No images found in the database.", 404
        # Process each image sequentially
        results = []
        for image_path, image_blob, existing_tags in images:
            try:
                # Log which image is being processed
                print(f"Processing image: {image_path}")
                # Ensure image_blob is in bytes format
                if isinstance(image_blob, memoryview):
                    image_blob = image_blob.tobytes()
                # Validate file extension
                if not allowed_file(image_path):
                    print(f"Skipping invalid file type: {image_path}")
                    continue
                # Convert .webp to .jpg if necessary
                if image_path.lower().endswith(".webp"):
                    image = Image.open(BytesIO(image_blob))
                    buffered = BytesIO()
                    image.convert("RGB").save(buffered, format="JPEG")
                    image_blob = buffered.getvalue()
                # Step 1: Detect labels using AWS Rekognition
                rekognition_response = rekognition.detect_labels(
                    Image={"Bytes": image_blob}, MaxLabels=20, MinConfidence=50
                )
                rekognition_labels = rekognition_response.get("Labels", [])
                rekognition_labels_with_confidence = [
                    (label["Name"], label["Confidence"]) for label in rekognition_labels
                ]
                # Step 2: Detect labels using Qwen
                qwen_labels = generate_labels_with_qwen(image_blob)
                # Step 3: Combine labels from AWS Rekognition and Qwen
                combined_labels = rekognition_labels_with_confidence + [
                    (label["Name"], label["Confidence"]) for label in qwen_labels
                ]
                # Deduplicate combined labels
                combined_labels_dict = {}
                for label, confidence in combined_labels:
                    if (
                        label not in combined_labels_dict
                        or confidence > combined_labels_dict[label]
                    ):
                        combined_labels_dict[label] = confidence
                combined_labels_with_confidence = list(combined_labels_dict.items())
                # Step 4: Add predefined tags to the combined labels
                if existing_tags:
                    try:
                        existing_tags_list = json.loads(
                            existing_tags
                        )  # Deserialize JSON string
                    except (TypeError, json.JSONDecodeError):
                        existing_tags_list = existing_tags  # Assume it's already a list
                    for tag in existing_tags_list:
                        combined_labels_with_confidence.append(
                            (tag, 100.0)
                        )  # Predefined tags get 100% confidence
                # Sort labels by confidence scores in descending order
                combined_labels_with_confidence.sort(key=lambda x: x[1], reverse=True)
                # Step 5: Generate meta tags using Ollama
                meta_title, meta_description = generate_meta_tags(
                    combined_labels_with_confidence
                )
                # Step 6: Update the database with labels, meta title, meta description, and confidence scores
                update_image_in_postgres(
                    image_path,
                    combined_labels_with_confidence,  # Combined labels with confidence scores
                    meta_title,
                    meta_description,
                )
                # Log success message
                print(f"Successfully processed image: {image_path}")
                # Add result for this image
                results.append(
                    {
                        "image_path": image_path,
                        "labels": [
                            label for label, _ in combined_labels_with_confidence
                        ],
                        "meta_title": meta_title,
                        "meta_description": meta_description,
                        "confidence_scores": {
                            label: score
                            for label, score in combined_labels_with_confidence
                        },
                    }
                )
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                results.append(
                    {
                        "image_path": image_path,
                        "error": str(e),
                    }
                )
        # Return a summary of the results
        return jsonify(results), 200
    except Exception as e:
        error_message = f"Error processing all images: {str(e)}"
        print(error_message)
        return error_message, 500


@app.route("/")
def start_page():
    return render_template("index1.html")


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Optionally, trigger the automation process on startup
    with app.app_context():
        print("Automatically processing all images...")
        with app.test_client() as client:
            response = client.get("/process_all_images")
            print(response.json)  # Print the results of the automation process
    app.run(port=5001)
