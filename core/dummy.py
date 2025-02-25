import os
import json
from azure.storage.blob import BlobServiceClient
import psycopg2

# Azure Storage Account Details
STORAGE_ACCOUNT_NAME = ""
STORAGE_ACCOUNT_KEY = ""
CONTAINER_NAME = ""

# PostgreSQL Database Connection Details
DB_HOST = ""
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = ""

# Predefined tags for image paths (blob names)
tags_mapping = {
    "2025/01/9_GCCs-Empowering-the-Future-of.jpg": [
        "Handshake",
        "Formal Conversation",
        "Suitcase",
    ],
    "2025/01/Logo-White-1.webp": ["Pixcy", "Logo"],
    "2025/01/Privacy-policy-1.jpg": ["Privacy Policy", "Security", "Network"],
    "2025/01/TermsConditions.jpg": ["Signature", "Notebook", "Writing"],
    "2025/01/favicon-1.png": ["Pixcy", "Logo", "Favicon"],
    "2025/01/phone-612061_1280-1.jpg": ["Logo", "Message", "Calling"],
    "2025/01/phone-612061_1280-1024x726.jpg": ["Logo", "Message", "Calling"],
    "2025/01/phone-612061_1280-150x150.jpg": ["Logo", "Message", "Calling"],
    "2025/01/phone-612061_1280-300x213.jpg": ["Logo", "Message", "Calling"],
    "2025/01/phone-612061_1280-768x544.jpg": ["Logo", "Message", "Calling"],
    "2025/01/phone-612061_1280.jpg": ["Logo", "Message", "Calling"],
    "2025/02/mt-sample-background-1024x683.jpg": [
        "Ice Mountains",
        "Parachute",
        "Stars",
    ],
    "2025/02/mt-sample-background-150x150.jpg": ["Ice Mountains", "Parachute", "Stars"],
    "2025/02/mt-sample-background-1536x1024.jpg": [
        "Ice Mountains",
        "Parachute",
        "Stars",
    ],
    "2025/02/mt-sample-background-300x200.jpg": ["Ice Mountains", "Parachute", "Stars"],
    "2025/02/mt-sample-background-768x512.jpg": ["Ice Mountains", "Parachute", "Stars"],
    "2025/02/mt-sample-background.jpg": ["Ice Mountains", "Parachute", "Stars"],
    "Picxy_Dummy3/39057dac4edcfac1efc0f1104d770b6f.jpg": [
        "Trains",
        "Railway Station",
        "Tracks",
    ],
    "Picxy_Dummy3/76180921bd77fb6fc8441fcda2a9bf22.jpg": [
        "Shopping",
        "Women",
        "Paper Bags",
    ],
    "Picxy_Dummy3/9a1ba1a268a61ad8c7609dabd02b596b.jpg": [
        "Sculptures",
        "Market",
        "God Statues",
        "Shopkeeper",
    ],
    "Picxy_Dummy3/c5c027ff1b0e965f6ccc54570211719e.jpg": [
        "Vehicles",
        "Night",
        "Highway",
    ],
    "Pixcy_Dummy/Pawan Kalyan.jpg": [
        "Pawan Kalyan",
        "Deputy Chief Minister",
        "Andhra Pradesh",
    ],
    "Pixcy_Dummy/Pongal-Celebration-pixcy-OS835478.jpg": [
        "Pongal Celebrations",
        "Buffalo",
        "Balloons",
        "Children",
        "Adults",
    ],
    "Pixcy_Dummy/Young-Mother-pumping-breastmilk-to-the-baby-bottle-picxy-XH769183.jpg": [
        "Milk Feeding",
        "Baby",
        "Mother",
    ],
    "Pixcy_Dummy/biryani.webp": ["Biryani", "Food", "Rice", "Chicken"],
    "Pixcy_Dummy/black.jpg": ["Black Shirt", "Tshirt", "Joggers"],
    "Pixcy_Dummy/bouquet.webp": ["Bouquet", "Flowers"],
    "Pixcy_Dummy/candidpose.jpg": ["Candid", "Pose", "Child", "Boy"],
    "Pixcy_Dummy/chilling.webp": ["Popcorn", "Video Games", "Controllers", "Sofa"],
    "Pixcy_Dummy/christmas.webp": ["Christmas Decoration", "Shopping Mall"],
    "Pixcy_Dummy/family.webp": ["Candid", "Family", "Food"],
    "Pixcy_Dummy/food.webp": ["Food", "Noodles", "Chopsticks"],
    "Pixcy_Dummy/gautam-gambhir.webp": ["Gautam Gambhir", "Ball", "Gloves"],
    "Pixcy_Dummy/gym.webp": ["Gym", "Plank", "Treadmill"],
    "Pixcy_Dummy/holi.webp": ["Candid", "Holi", "Girl"],
    "Pixcy_Dummy/image1.jpg": ["Racer", "Sports Bike"],
    "Pixcy_Dummy/place.jpg": ["Beautiful"],
    "Pixcy_Dummy/portrait.webp": ["Portrait"],
    "Pixcy_Dummy/stock.webp": ["Stocks", "Indices", "Stock Market"],
    "Pixcy_Dummy/studio-portrait-of-seven-business-picture__bcp004-24.png": [
        "People",
        "Formal Wears",
    ],
    "Pixcy_Dummy2/birds-flying.jpg": ["Birds", "Water", "Lake"],
    "Pixcy_Dummy2/messy-office-table-top-objects-image.png": ["Spectacles", "Messy"],
    "hello.py": [],
}

# Create a BlobServiceClient
connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)


def fetch_and_store_images():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()
        # Ensure tags and confidence_score columns exist
        cursor.execute(
            """
            ALTER TABLE images 
            ADD COLUMN IF NOT EXISTS tags TEXT,
            ADD COLUMN IF NOT EXISTS confidence_score FLOAT DEFAULT 0.0;
        """
        )
        # Get the container client
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        print(f"Listing blobs in container '{CONTAINER_NAME}'...")
        blobs = container_client.list_blobs()
        for blob in blobs:
            blob_name = blob.name
            print(f"Found blob: {blob_name}")
            # Skip folders
            if blob_name.endswith("/"):
                print(f"Skipping folder: {blob_name}")
                continue
            # Extract image name from blob_name (remove folder paths)
            image_name = os.path.basename(blob_name)
            print(f"Extracted image name: {image_name}")
            # Download the blob data
            blob_client = container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()
            # Check if the image already exists in the database
            select_query = """
            SELECT tags FROM images WHERE image_path = %s;
            """
            cursor.execute(select_query, (blob_name,))
            existing_record = cursor.fetchone()
            if existing_record:
                # If the record exists, check if the tags column is already populated
                existing_tags = existing_record[0]
                if existing_tags:
                    print(
                        f"Tags already exist for image: {blob_name}. Skipping update."
                    )
                    continue
            # Get predefined tags or set to empty list, then serialize to JSON string
            tags_list = tags_mapping.get(blob_name, [])
            tags_json = json.dumps(tags_list)  # Convert list to JSON string
            # Insert image binary data into PostgreSQL along with image_name, tags, and confidence_score
            insert_query = """
            INSERT INTO images (
                image_path, 
                image_blob, 
                image_name, 
                "Meta-Title", 
                "Meta-Description", 
                status, 
                lat, 
                lon, 
                tags,
                confidence_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (image_path) DO UPDATE SET
                image_blob = EXCLUDED.image_blob,
                image_name = EXCLUDED.image_name,
                "Meta-Title" = EXCLUDED."Meta-Title",
                "Meta-Description" = EXCLUDED."Meta-Description",
                status = EXCLUDED.status,
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon,
                tags = EXCLUDED.tags,
                confidence_score = EXCLUDED.confidence_score;
            """
            cursor.execute(
                insert_query,
                (
                    blob_name,  # image_path
                    psycopg2.Binary(blob_data),  # image_blob
                    image_name,  # image_name
                    "Unknown Image",  # Default Meta-Title
                    "No description available.",  # Default Meta-Description
                    "D",  # Default status ('Draft')
                    0.0,  # Default latitude
                    0.0,  # Default longitude
                    tags_json,  # Serialized JSON string
                    0.0,  # Default confidence_score
                ),
            )
            print(f"Inserted image with path {blob_name} and tags: {tags_json}")
        # Commit changes and close the connection
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    fetch_and_store_images()
