
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    print("Supabase URL or key missing in .env")
else:
    try:
        supabase = create_client(url, key)
        print("Supabase client created successfully!")
    except Exception as e:
        print("Connection setup failed:", e)
