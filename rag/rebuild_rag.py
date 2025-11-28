import argparse
import logging
import os

# Google Cloud Libraries
from google.cloud import storage
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError

# --- CONFIGURATION & SETUP ---

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --- UTILITY FUNCTIONS ---


def upload_file_to_gcs(
    bucket_name: str, source_file_path: str, destination_blob_name: str
):
    """Uploads a file to Google Cloud Storage (GCS)."""
    logger.info(
        f"Uploading {source_file_path} to gs://{bucket_name}/{destination_blob_name}..."
    )

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Upload the file from the local path
        blob.upload_from_filename(source_file_path)

        logger.info(
            f"✅ Upload successful. GCS URI: gs://{bucket_name}/{destination_blob_name}"
        )
        return f"gs://{bucket_name}/{destination_blob_name}"

    except Exception as e:
        logger.error(f"❌ GCS Upload Failed: {e}")
        raise


def purge_documents(project_id: str, search_location: str, data_store_id: str):
    """Removes all documents from the specified Vertex AI Search data store."""
    logger.info(
        f"Initiating purge for Data Store: {data_store_id} in {search_location}..."
    )

    # 1. Setup client with correct regional endpoint
    client_options = ClientOptions(
        api_endpoint=f"{search_location}-discoveryengine.googleapis.com"
    )
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    # 2. Define the parent resource path
    parent_path = client.branch_path(
        project=project_id,
        location=search_location,
        data_store=data_store_id,
        branch="default_branch",
    )

    # 3. Construct the Purge Request
    request = discoveryengine.PurgeDocumentsRequest(
        parent=parent_path,
        filter="*",  # Required to purge ALL documents
        force=True,  # Actually perform the deletion
    )

    # 4. Run the request
    operation = client.purge_documents(request=request)

    logger.info(
        f"⏳ Purge operation started: {operation.operation.name}. Waiting for completion..."
    )

    # 5. Wait for the operation to complete
    operation.result()

    logger.info("✅ Purge successful. Data Store is now empty.")


def import_documents(
    project_id: str, search_location: str, data_store_id: str, gcs_uri: str
):
    """Imports documents from a GCS URI to the specified data store."""
    logger.info(f"Initiating import from {gcs_uri} to Data Store: {data_store_id}...")

    # 1. Setup client with correct regional endpoint
    client_options = ClientOptions(
        api_endpoint=f"{search_location}-discoveryengine.googleapis.com"
    )
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    # 2. Define the parent resource path
    parent_path = client.branch_path(
        project=project_id,
        location=search_location,
        data_store=data_store_id,
        branch="default_branch",
    )

    # 3. Construct the Import Request
    request = discoveryengine.ImportDocumentsRequest(
        parent=parent_path,
        gcs_source=discoveryengine.GcsSource(
            input_uris=[gcs_uri],
            # data is JSONL, which is treated as structured
            data_schema="custom",
        ),
        auto_generate_ids=True,
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,  # FULL re-indexing
    )

    # 4. Run the request
    operation = client.import_documents(request=request)

    logger.info(
        f"⏳ Import operation started: {operation.operation.name}. Waiting for completion..."
    )

    # 5. Wait for the operation to complete
    operation.result()

    logger.info(f"✅ Import successful. {gcs_uri} is now indexed.")


# --- MAIN EXECUTION LOGIC ---


def rebuild_rag_database(
    data_store_id: str,
    bucket: str,
    source_file: str,
    project_id: str,
    search_location: str,
):

    logger.info("--- STARTING RAG DATABASE REBUILD SCRIPT ---")

    # 1. Check if source file exists locally
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found at local path: {source_file}")

    # The GCS object name will be the same as the local file name
    gcs_object_name = os.path.basename(source_file)

    # --- STEP 1: UPLOAD NEW FILE TO GCS ---
    gcs_uri = upload_file_to_gcs(
        bucket_name=bucket,
        source_file_path=source_file,
        destination_blob_name=gcs_object_name,
    )

    # --- STEP 2: PURGE EXISTING DATA ---
    try:
        purge_documents(
            project_id=project_id,
            search_location=search_location,
            data_store_id=data_store_id,
        )
    except GoogleAPICallError as e:
        logger.error(
            f"❌ Purge failed for data store {data_store_id}. Ensure Data Store ID and search_location are correct."
        )
        logger.error(f"Details: {e}")
        # Terminate if purge fails, as re-indexing would mix old and new data
        return

    # --- STEP 3: IMPORT/REINDEX NEW DATA ---
    import_documents(
        project_id=project_id,
        search_location=search_location,
        data_store_id=data_store_id,
        gcs_uri=gcs_uri,
    )

    logger.info("--- RAG DATABASE REBUILD COMPLETE ---")


# --- ARGUMENT PARSING ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically purges and re-indexes a Vertex AI Search Data Store from a new JSONL file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--bucket",
        type=str,
        default="rag-based-video-games-bot",
        help="The ID of the GCS bucket where the file is stored.",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        required=True,
        help="REQUIRED: Local path to the new JSONL file (e.g., filtered_QA_Video_Games_new.jsonl).",
    )

    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env if present

    # Get necessary environment variables
    PROJECT_ID = os.getenv("PROJECT_ID")
    SEARCH_LOCATION = os.getenv("SEARCH_REGION")
    DATA_STORE_ID = os.getenv("DATA_STORE_ID")

    if not PROJECT_ID or not SEARCH_LOCATION or not DATA_STORE_ID:
        logger.error(
            "FATAL: Please set the PROJECT_ID, SEARCH_REGION, and DATA_STORE_ID environment variables."
        )
        exit(1)

    # Run the main rebuild function
    rebuild_rag_database(
        data_store_id=DATA_STORE_ID,
        bucket=args.bucket,
        source_file=args.source_file,
        project_id=PROJECT_ID,
        search_location=SEARCH_LOCATION,
    )
