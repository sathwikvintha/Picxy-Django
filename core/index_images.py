import psycopg2
import json
from elasticsearch import Elasticsearch

# PostgreSQL Database Configuration
DB_HOST = "localhost"
DB_NAME = "pixcy"
DB_USER = "postgres"
DB_PASSWORD = "Sathwik@20022004"
# Elasticsearch Configuration
ES_HOST = "http://localhost:9200"  # Updated to match Dockerized Elasticsearch
ES_INDEX = "images_index"
# Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL!")
except Exception as e:
    print(f"❌ PostgreSQL connection error: {e}")
    exit()
# Connect to Elasticsearch
try:
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ValueError("Elasticsearch connection failed!")
    print("✅ Connected to Elasticsearch!")
except Exception as e:
    print(f"❌ Elasticsearch connection error: {e}")
    exit()
# Fetch all image IDs from PostgreSQL
try:
    cursor.execute("SELECT image_id FROM images;")
    db_image_ids = {
        row[0] for row in cursor.fetchall()
    }  # Set of image IDs in PostgreSQL
    print(f"✅ Fetched {len(db_image_ids)} image IDs from PostgreSQL.")
except Exception as e:
    print(f"❌ Error fetching image IDs from PostgreSQL: {e}")
    exit()
# Fetch all image IDs from Elasticsearch
try:
    es_response = es.search(
        index=ES_INDEX, body={"query": {"match_all": {}}}, size=10000
    )
    es_image_ids = {
        hit["_id"] for hit in es_response["hits"]["hits"]
    }  # Set of image IDs in Elasticsearch
    print(f"✅ Fetched {len(es_image_ids)} image IDs from Elasticsearch.")
except Exception as e:
    print(f"❌ Error fetching image IDs from Elasticsearch: {e}")
    exit()
# Identify image IDs to delete (present in Elasticsearch but not in PostgreSQL)
ids_to_delete = es_image_ids - db_image_ids
print(f"ℹ️ Found {len(ids_to_delete)} images to delete from Elasticsearch.")
# Delete images from Elasticsearch that are not in PostgreSQL
for image_id in ids_to_delete:
    try:
        es.delete(index=ES_INDEX, id=image_id)
        print(f"✅ Deleted image_id {image_id} from Elasticsearch.")
    except Exception as e:
        print(f"❌ Error deleting image_id {image_id} from Elasticsearch: {e}")
# Fetch images data from PostgreSQL
try:
    cursor.execute("SELECT * FROM images;")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    print(f"✅ Fetched {len(rows)} rows from PostgreSQL.")
except Exception as e:
    print(f"❌ Error fetching data from PostgreSQL: {e}")
    exit()
# Index data into Elasticsearch
for row in rows:
    try:
        image_data = dict(zip(columns, row))
        # Remove binary data (Elasticsearch does not support bytea)
        if isinstance(image_data.get("image_blob"), memoryview):
            image_data["image_blob"] = None
        # Convert JSONB fields to lists/dictionaries
        json_fields = ["labels", "tags", "vision_labels", "confidence_scores"]
        for field in json_fields:
            if image_data.get(field) is None:
                image_data[field] = []  # Default empty list
            elif isinstance(image_data[field], str):
                try:
                    image_data[field] = json.loads(image_data[field])
                except json.JSONDecodeError:
                    image_data[field] = []
        # Ensure Meta-Title and Meta-Description have default values
        image_data["Meta-Title"] = image_data.get("Meta-Title") or "Unknown Image"
        image_data["Meta-Description"] = (
            image_data.get("Meta-Description") or "No description available."
        )
        # Ensure labels contain valid strings (not null or empty)
        if isinstance(image_data["labels"], list):
            image_data["labels"] = [
                label.strip()
                for label in image_data["labels"]
                if isinstance(label, str) and label.strip()
            ]
        else:
            image_data["labels"] = []
        # Debug: Print labels before indexing
        print(f"Labels for image_id {image_data['image_id']}: {image_data['labels']}")
        # Skip indexing if labels are empty
        if not image_data["labels"]:
            print(
                f"❌ Skipping image_id {image_data['image_id']} because labels are empty/null."
            )
            continue
        # Convert confidence_score to float
        image_data["confidence_score"] = float(
            image_data.get("confidence_score") or 0.0
        )
        # Convert lat/lon to float
        image_data["lat"] = float(image_data.get("lat") or 0.0)
        image_data["lon"] = float(image_data.get("lon") or 0.0)
        # Set status default
        image_data["status"] = image_data.get("status") or "D"
        # Index document into Elasticsearch
        try:
            es.index(index=ES_INDEX, id=image_data["image_id"], document=image_data)
            print(f"✅ Indexed image_id {image_data['image_id']} successfully.")
        except Exception as e:
            print(f"❌ Error indexing image_id {image_data['image_id']}: {e}")
    except Exception as e:
        print(f"❌ Error processing row {row}: {e}")
print("✅ Successfully indexed all valid images into Elasticsearch!")
# Close connections
cursor.close()
conn.close()
