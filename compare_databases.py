"""
Database Comparison Tool
Compare old Gemini-based database with new universal database
"""
import os
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*80)
print("  DATABASE COMPARISON TOOL")
print("="*80)

# Check old database
print("\n1️⃣  OLD DATABASE (Gemini-based)")
print("-" * 80)
old_db_path = os.path.join(BASE_DIR, "legal_db")
if os.path.exists(old_db_path):
    try:
        old_client = chromadb.PersistentClient(path=old_db_path)
        old_collection = old_client.get_collection("legal_brain")
        old_count = old_collection.count()
        
        print(f"✅ Status: EXISTS")
        print(f"📍 Location: {old_db_path}")
        print(f"📦 Collection: legal_brain")
        print(f"📊 Chunks: {old_count:,}")
        print(f"🔧 Embedding: Gemini text-embedding-004 (768 dimensions)")
        print(f"🔑 API Required: YES (for search)")
        print(f"🌐 Offline Search: NO")
    except Exception as e:
        print(f"❌ Status: EXISTS but cannot connect")
        print(f"❌ Error: {e}")
else:
    print(f"❌ Status: NOT FOUND")
    print(f"📍 Expected location: {old_db_path}")

# Check new universal database
print("\n2️⃣  NEW DATABASE (Universal)")
print("-" * 80)
new_db_path = os.path.join(BASE_DIR, "legal_db_universal")
if os.path.exists(new_db_path):
    try:
        new_client = chromadb.PersistentClient(path=new_db_path)
        new_collection = new_client.get_collection("legal_brain_universal")
        new_count = new_collection.count()
        
        print(f"✅ Status: EXISTS")
        print(f"📍 Location: {new_db_path}")
        print(f"📦 Collection: legal_brain_universal")
        print(f"📊 Chunks: {new_count:,}")
        print(f"🔧 Embedding: ChromaDB all-MiniLM-L6-v2 (384 dimensions)")
        print(f"🔑 API Required: NO")
        print(f"🌐 Offline Search: YES")
    except Exception as e:
        print(f"❌ Status: EXISTS but cannot connect")
        print(f"❌ Error: {e}")
else:
    print(f"❌ Status: NOT FOUND")
    print(f"📍 Expected location: {new_db_path}")
    print(f"\n⚠️  To create universal database, run:")
    print(f"   python build_database_universal.py")

# Check which database app.py is using
print("\n3️⃣  CURRENT APP CONFIGURATION")
print("-" * 80)
try:
    with open(os.path.join(BASE_DIR, "app.py"), 'r', encoding='utf-8') as f:
        app_content = f.read()
        
    if 'legal_db_universal' in app_content:
        print("✅ App is configured to use: UNIVERSAL DATABASE")
        print("📦 Database: legal_db_universal")
        print("📦 Collection: legal_brain_universal")
    elif 'legal_db"' in app_content or 'legal_db/' in app_content:
        print("⚠️  App is configured to use: OLD DATABASE")
        print("📦 Database: legal_db")
        print("📦 Collection: legal_brain")
        print("\n💡 To switch to universal database:")
        print("   1. Make sure universal database is built")
        print("   2. Update DB_PATH in app.py to 'legal_db_universal'")
        print("   3. Update collection name to 'legal_brain_universal'")
    else:
        print("❓ Could not determine database configuration")
except Exception as e:
    print(f"❌ Error reading app.py: {e}")

# Summary
print("\n" + "="*80)
print("  SUMMARY")
print("="*80)

if os.path.exists(old_db_path) and os.path.exists(new_db_path):
    print("✅ Both databases exist - you can switch between them")
    print("\n📝 To use OLD database:")
    print("   copy app_gemini_backup.py app.py")
    print("\n📝 To use NEW database:")
    print("   (Already configured if you see 'legal_db_universal' above)")
elif os.path.exists(old_db_path) and not os.path.exists(new_db_path):
    print("⚠️  Only old database exists")
    print("\n📝 To create universal database:")
    print("   python build_database_universal.py")
elif not os.path.exists(old_db_path) and os.path.exists(new_db_path):
    print("✅ Universal database exists and ready to use")
else:
    print("❌ No databases found")
    print("\n📝 To create universal database:")
    print("   python build_database_universal.py")

print("\n" + "="*80)
