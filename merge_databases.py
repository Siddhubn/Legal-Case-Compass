import os
import chromadb

# --- CONFIGURATION ---

# 1. Set the path to your main "master" database
MASTER_DB_PATH = "C:\\LCC\\DATA\\legal_db"

# 2. List all the "source" databases you want to merge *from*
#    (Use the names you created in Step 1)
SOURCE_DB_PATHS = [
    "C:\\LCC\\DATA\\legal_db_1990_2000",
    "C:\\LCC\\DATA\\legal_db_2000_2010",
    "C:\\LCC\\DATA\\legal_db_2020_2025"
]

# 3. The collection name (must match what we used in build_database.py)
COLLECTION_NAME = "legal_brain"

# 4. How many items to move at a time (500 is a safe number)
BATCH_SIZE = 500

# --- MERGE SCRIPT ---

def main():
    # 1. Connect to the master database
    print(f"Connecting to MASTER database at: {MASTER_DB_PATH}")
    try:
        master_client = chromadb.PersistentClient(path=MASTER_DB_PATH)
        master_collection = master_client.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"FATAL: Could not open master database. Error: {e}")
        return

    print(f"Master DB currently has {master_collection.count()} items.")
    print("---" * 10)

    # 2. Loop through each source database
    for source_path in SOURCE_DB_PATHS:
        print(f"\nProcessing SOURCE database: {source_path}")
        try:
            source_client = chromadb.PersistentClient(path=source_path)
            source_collection = source_client.get_collection(COLLECTION_NAME)
            
            total_items = source_collection.count()
            if total_items == 0:
                print("  > This database is empty. Skipping.")
                continue
                
            print(f"  > Found {total_items} items to merge.")

            # 3. Get all data from the source in batches
            for offset in range(0, total_items, BATCH_SIZE):
                print(f"    > Merging batch {offset} to {offset + BATCH_SIZE}...")
                
                # Get a batch of data from the source
                batch_data = source_collection.get(
                    limit=BATCH_SIZE,
                    offset=offset,
                    include=["metadatas", "documents", "embeddings"] # Get everything
                )
                
                # 4. Write that batch to the master database
                #    'upsert' = "update if exists, insert if new"
                #    This is safe and prevents duplicates.
                master_collection.upsert(
                    ids=batch_data['ids'],
                    embeddings=batch_data['embeddings'],
                    documents=batch_data['documents'],
                    metadatas=batch_data['metadatas']
                )
            
            print(f"  > Successfully merged {total_items} items from {source_path}")

        except Exception as e:
            print(f"  > ERROR processing {source_path}: {e}")
            print("  > Skipping this database and continuing...")

    print("\n---" * 10)
    print("✅✅ MERGE COMPLETE! ✅✅")
    print(f"Master DB now has {master_collection.count()} total items.")
    print("You can now safely delete the source folders (e.g., 'legal_db_1990_2000').")

if __name__ == "__main__":
    main()