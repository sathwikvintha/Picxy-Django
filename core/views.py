import os
import csv
import boto3
import json
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file
from celery import Celery
from django.shortcuts import render, redirect, get_object_or_404
import psycopg2
from io import BytesIO
import base64
import ollama
import dashscope
from django.http import HttpResponse, Http404, JsonResponse
from django.conf import settings
from .models import Image
from django.views.decorators.csrf import csrf_exempt
from elasticsearch import Elasticsearch
from django.apps import apps
from django.db import connection
from PIL import Image as PILImage


es = Elasticsearch("http://localhost:9200")

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

DB_HOST = "localhost"
DB_NAME = "pixcy"
DB_USER = "postgres"
DB_PASSWORD = "Sathwik@20022004"


# hello
# Load AWS Credentials from CSV
def load_aws_credentials(filename="aws_credentials.csv"):
    try:
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader)
            aws_access_key, aws_secret_key = next(reader)
            return aws_access_key, aws_secret_key
    except Exception as e:
        print(f"Error loading AWS credentials: {e}")
        return None, None


AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = load_aws_credentials()
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    rekognition = boto3.client(
        "rekognition",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )
else:
    raise ValueError("AWS credentials not found! Check 'aws_credentials.csv'.")

dashscope.api_key = "your_qwen_api_key"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def fetch_by_filename(request):
    filename = request.GET.get("filename", "").strip()
    if not filename:
        return JsonResponse({"error": "No filename provided"}, status=400)

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT image_name, image_path, labels, tags, "Meta-Title", "Meta-Description"
            FROM images
            WHERE image_name = %s
            """,
            [filename],
        )
        row = cursor.fetchone()

    if row:
        image_name, image_path, labels, tags, meta_title, meta_description = row

        labels = json.loads(labels) if labels else []
        tags = json.loads(tags) if tags else []

        print(f"Meta Title: {meta_title}")
        print(f"Meta Description: {meta_description}")

        return JsonResponse(
            {
                "image_name": image_name,
                "image_path": image_path,
                "labels": labels,
                "tags": tags,
                "meta_title": meta_title,
                "meta_description": meta_description,
            }
        )

    return JsonResponse({"error": "Image not found"}, status=404)


def fetch_all_images_from_postgres():
    """
    Fetch all images from PostgreSQL and return as a list.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT image_id, image_path, image_name, labels, tags, "Meta-Title", "Meta-Description", confidence_scores
            FROM images
        """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return rows
    except Exception as e:
        print(f"Error fetching images from PostgreSQL: {e}")
        return []


@csrf_exempt
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        try:
            uploaded_file = request.FILES["image"]
            image_name = uploaded_file.name
            image_path = image_name

            if not allowed_file(image_name):
                return JsonResponse({"error": "Invalid file type."}, status=400)

            image_blob = uploaded_file.read()

            if image_name.lower().endswith(".webp"):
                image = PILImage.open(BytesIO(image_blob))
                buffered = BytesIO()
                image.convert("RGB").save(buffered, format="JPEG")
                image_blob = buffered.getvalue()
                image_name = image_name.rsplit(".", 1)[0] + ".jpg"
                image_path = image_name

            conn = psycopg2.connect(
                host="localhost",
                database="pixcy",
                user="postgres",
                password="Sathwik@20022004",
            )
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO images (image_name, image_path, image_blob, labels, "Meta-Title", "Meta-Description")
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING image_id;
                """,
                (
                    image_name,
                    image_path,
                    psycopg2.Binary(image_blob),
                    json.dumps([]),
                    "Unknown Image",
                    "No description available.",
                ),
            )
            image_id = cursor.fetchone()[0]
            conn.commit()

            rekognition_response = rekognition.detect_labels(
                Image={"Bytes": image_blob}, MaxLabels=20, MinConfidence=50
            )
            rekognition_labels = rekognition_response.get("Labels", [])
            rekognition_labels_with_confidence = [
                (label["Name"], label["Confidence"]) for label in rekognition_labels
            ]

            qwen_labels = generate_labels_with_qwen(image_blob)
            qwen_labels_with_confidence = [
                (label["Name"], label["Confidence"]) for label in qwen_labels
            ]

            combined_labels_dict = {}
            for label, confidence in (
                rekognition_labels_with_confidence + qwen_labels_with_confidence
            ):
                if (
                    label not in combined_labels_dict
                    or confidence > combined_labels_dict[label]
                ):
                    combined_labels_dict[label] = confidence
            combined_labels_with_confidence = list(combined_labels_dict.items())

            cursor.execute("SELECT tags FROM images WHERE image_id = %s;", (image_id,))
            existing_tags = cursor.fetchone()[0]
            if existing_tags:
                try:
                    existing_tags_list = json.loads(existing_tags)
                except (TypeError, json.JSONDecodeError):
                    existing_tags_list = existing_tags
                for tag in existing_tags_list:
                    combined_labels_with_confidence.append((tag, 100.0))

            meta_title, meta_description = generate_meta_tags(
                combined_labels_with_confidence
            )

            cursor.execute(
                """
                UPDATE images
                SET labels = %s, "Meta-Title" = %s, "Meta-Description" = %s
                WHERE image_id = %s;
                """,
                (
                    json.dumps([label for label, _ in combined_labels_with_confidence]),
                    meta_title or "Unknown Image",
                    meta_description or "No description available.",
                    image_id,
                ),
            )
            conn.commit()

            cursor.close()
            conn.close()

            return JsonResponse(
                {
                    "message": "Image uploaded and processed successfully.",
                    "image_id": image_id,
                    "image_path": image_path,
                    "labels": [label for label, _ in combined_labels_with_confidence],
                    "meta_title": meta_title or "Unknown Image",
                    "meta_description": meta_description or "No description available.",
                },
                status=200,
            )

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request."}, status=400)


def update_image_in_postgres(
    image_path, labels_with_confidence, meta_title, meta_description
):
    """
    Updates an image's metadata in PostgreSQL and Elasticsearch.
    """
    try:
        # Debug: Print received data
        print("\n--- Debugging Before Processing ---")
        print(f"Image Path: {image_path}")
        print(f"Raw Labels with Confidence: {labels_with_confidence}")

        # Validate `labels_with_confidence`
        if not isinstance(labels_with_confidence, list) or not all(
            isinstance(i, tuple) and len(i) == 2 for i in labels_with_confidence
        ):
            raise ValueError(
                "labels_with_confidence must be a list of (label, confidence) tuples."
            )

        # Convert labels to JSONB format (list of strings)
        labels_jsonb = json.dumps([label for label, _ in labels_with_confidence])

        # Convert confidence scores to JSONB format (dictionary)
        confidence_scores_jsonb = json.dumps(
            {label: score for label, score in labels_with_confidence}
        )

        # Debug: Print formatted JSONB values
        print("\n--- JSONB Data to be Stored ---")
        print(f"Labels JSONB: {labels_jsonb}")
        print(f"Confidence Scores JSONB: {confidence_scores_jsonb}")
        print(f"Meta Title: {meta_title}")
        print(f"Meta Description: {meta_description}")

        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Update query for JSONB columns
        update_query = """
        UPDATE images
        SET labels = %s::jsonb, 
        "Meta-Title" = %s, 
        "Meta-Description" = %s, 
        confidence_scores = %s::jsonb
        WHERE image_path = %s;
        """
        cursor.execute(
            update_query,
            (
                labels_jsonb,
                meta_title,
                meta_description,
                confidence_scores_jsonb,
                image_path,
            ),
        )

        # Commit the transaction
        conn.commit()
        cursor.close()
        conn.close()
        print(f"\n✅ Successfully updated metadata for image: {image_path}")

        # Update Elasticsearch
        update_image_in_elasticsearch(
            image_path=image_path,
            labels_with_confidence=labels_with_confidence,
            meta_title=meta_title,
            meta_description=meta_description,
        )

    except Exception as e:
        print(f"\n❌ Error updating image in PostgreSQL: {e}")


@app.route("/process_all_images", methods=["GET"])
def process_all_images():
    try:
        # Fetch all images from the database
        images = fetch_all_images_from_postgres()
        if not images:
            return "No images found in the database.", 404

        results = []
        for image_path, image_blob, existing_tags in images:
            try:
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
                qwen_labels_with_confidence = [
                    (label["Name"], label.get("Confidence", 0.0))
                    for label in qwen_labels
                ]

                # Step 3: Combine labels from AWS Rekognition and Qwen
                combined_labels = (
                    rekognition_labels_with_confidence + qwen_labels_with_confidence
                )

                # Deduplicate combined labels
                combined_labels_dict = {}
                for label, confidence in combined_labels:
                    if (
                        label not in combined_labels_dict
                        or confidence > combined_labels_dict[label]
                    ):
                        combined_labels_dict[label] = confidence
                combined_labels_with_confidence = list(combined_labels_dict.items())

                # Debugging: Print labels with confidence
                print(f"Labels with confidence: {combined_labels_with_confidence}")

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
                results.append({"image_path": image_path, "error": str(e)})

        # Return a summary of the results
        return jsonify(results), 200
    except Exception as e:
        error_message = f"Error processing all images: {str(e)}"
        print(error_message)
        return error_message, 500


def process_single_image(image_path, image_blob):
    """
    Process a single image using AWS Rekognition, Qwen, and Ollama.
    """
    try:
        print(f"Processing image: {image_path}")

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

        # Step 4: Generate meta tags using Ollama
        meta_title, meta_description = generate_meta_tags(
            combined_labels_with_confidence
        )

        # Step 5: Update the database with labels, meta title, meta description, and confidence scores
        update_image_in_postgres(
            image_path,
            combined_labels_with_confidence,  # Combined labels with confidence scores
            meta_title,
            meta_description,
        )

        print(f"Successfully processed image: {image_path}")

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")


def update_image_in_elasticsearch(
    image_path, labels_with_confidence, meta_title, meta_description
):
    """
    Update an image's metadata in Elasticsearch.
    """
    try:
        # Fetch the image ID from PostgreSQL using image_path
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT image_id FROM images WHERE image_path = %s
            """,
            (image_path,),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise ValueError(f"No image found with path: {image_path}")

        image_id = row[0]

        # Convert labels and tags to nested format
        labels_nested = [
            {"label": label, "confidence": score}
            for label, score in labels_with_confidence
        ]

        # Create Elasticsearch document
        doc = {
            "image_id": image_id,
            "image_path": image_path,
            "labels": labels_nested,
            "meta_title": meta_title,
            "meta_description": meta_description,
            "confidence_scores": {
                label: score for label, score in labels_with_confidence
            },
        }

        # Index the document
        es.index(index="images_index", id=image_id, document=doc)
        print(f"✅ Successfully updated image in Elasticsearch: {image_path}")
    except Exception as e:
        print(f"❌ Error updating image in Elasticsearch: {e}")


def detect(request):
    if request.method == "POST":
        filename = request.POST.get("filename")
        max_labels = int(request.POST.get("label", 5))
        try:
            image = Image.objects.get(image_path=filename)
            # Simulate label detection (replace with actual logic)
            labels = image.labels[:max_labels]
            return render(
                request,
                "index.html",
                {
                    "filename": image.image_path,
                    "labels": labels,
                },
            )
        except Image.DoesNotExist:
            return render(request, "index.html", {"error": "Image not found."})
    return render(request, "index.html")


def list_images(request):
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host="localhost",
            database="pixcy",  # Replace with your database name
            user="postgres",  # Replace with your database username
            password="Sathwik@20022004",  # Replace with your database password
        )
        cursor = conn.cursor()

        # Fetch all images from the database
        cursor.execute(
            """
            SELECT image_id, image_path, image_blob, labels, "Meta-Title", "Meta-Description", confidence_scores
            FROM images
            """
        )
        rows = cursor.fetchall()

        # Prepare data for rendering
        images = []
        for row in rows:
            (
                image_id,
                image_path,
                image_blob,
                labels,
                meta_title,
                meta_description,
                confidence_scores,
            ) = row

            # Convert binary image data to Base64 for preview
            image_preview = (
                base64.b64encode(image_blob).decode("utf-8") if image_blob else None
            )

            # Ensure labels is a list
            if isinstance(labels, str):
                try:
                    labels = json.loads(labels)
                except (TypeError, json.JSONDecodeError):
                    labels = []

            # Ensure confidence_scores is a dictionary
            if isinstance(confidence_scores, str):
                try:
                    confidence_scores = json.loads(confidence_scores)
                except (TypeError, json.JSONDecodeError):
                    confidence_scores = {}

            images.append(
                {
                    "image_id": image_id,  # Ensure image_id matches images.html
                    "image_path": image_path,
                    "image_preview": image_preview,
                    "labels": labels,  # Ensure labels is always a list
                    "meta_title": meta_title,
                    "meta_description": meta_description,
                    "confidence_scores": confidence_scores,  # Add confidence_scores
                }
            )

        cursor.close()
        conn.close()

        # Debugging: Print image data in logs
        print("Fetched Images:", images)

        return render(request, "images.html", {"images": images})

    except Exception as e:
        print(f"Error in list_images: {str(e)}")
        return render(request, "images.html", {"error": str(e)})


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
            if any(conf >= 90 for _, conf in labels_with_confidence):
                filtered_labels.append(label)
        elif 55 <= confidence < 75:
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


def delete_image(request, image_id):
    """
    Delete an image from the database by its ID.
    """
    if request.method == "DELETE":
        try:
            if not image_id:
                return JsonResponse(
                    {"success": False, "error": "Missing image ID."}, status=400
                )

            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host="localhost",
                database="pixcy",
                user="postgres",
                password="Sathwik@20022004",
            )
            cursor = conn.cursor()

            # Check if the image exists before deleting
            cursor.execute(
                "SELECT image_id FROM images WHERE image_id = %s", (image_id,)
            )
            if not cursor.fetchone():
                cursor.close()
                conn.close()
                return JsonResponse(
                    {"success": False, "error": "Image not found."}, status=404
                )

            # Delete the image
            cursor.execute("DELETE FROM images WHERE image_id = %s", (image_id,))
            conn.commit()
            cursor.close()
            conn.close()

            return JsonResponse(
                {"success": True, "message": "Image deleted successfully."}, status=200
            )

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "error": "Invalid request method."}, status=400
    )


def search_images(request):
    if request.method == "GET":
        try:
            # Get the search query from the request
            query = request.GET.get("q")
            if not query:
                return JsonResponse(
                    {"error": "Please provide a search query."}, status=400
                )

            print(f"Searching for: {query}")

            # Construct the Elasticsearch query
            search_query = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"labels": query}},  # Match labels
                            {"match": {"tags": query}},  # Match tags
                            {"match": {"Meta-Title": query}},  # Match Meta-Title
                            {
                                "match": {"Meta-Description": query}
                            },  # Match Meta-Description
                        ],
                        "minimum_should_match": 1,
                    }
                }
            }

            # Execute the query
            response = es.search(index="images_index", body=search_query)

            # Extract matching results
            hits = response.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                source = hit["_source"]
                image_id = source.get("image_id")

                # Fetch the image_blob from PostgreSQL
                conn = psycopg2.connect(
                    host=DB_HOST,
                    database=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                )
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT image_blob FROM images WHERE image_id = %s;
                    """,
                    (image_id,),
                )
                row = cursor.fetchone()
                cursor.close()
                conn.close()

                # Convert binary image data to Base64 for preview
                image_blob = row[0] if row else None
                image_preview = (
                    base64.b64encode(image_blob).decode("utf-8") if image_blob else None
                )

                # Append result with image preview
                results.append(
                    {
                        "image_id": source.get("image_id"),
                        "image_name": source.get("image_name"),
                        "image_path": source.get("image_path"),
                        "labels": source.get("labels"),
                        "tags": source.get("tags"),
                        "meta_title": source.get("Meta-Title"),
                        "meta_description": source.get("Meta-Description"),
                        "image_preview": image_preview,  # Add Base64-encoded image
                    }
                )

            # Return the results
            return JsonResponse(results, safe=False, status=200)

        except Exception as e:
            print(f"Unexpected error during search: {str(e)}")
            return JsonResponse({"error": "An unexpected error occurred."}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=405)


@app.route("/")
def index(request):
    return render(request, "index.html")


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Optionally, trigger the automation process on startup
    print("Indexing images into Elasticsearch...")
    with app.app_context():
        print("Automatically processing all images...")
        with app.test_client() as client:
            response = client.get("/process_all_images")
            print(response.json)  # Print the results of the automation process

    app.run(port=5001)
