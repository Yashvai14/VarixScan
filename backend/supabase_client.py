import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration with flexible key handling
SUPABASE_URL = os.getenv("SUPABASE_URL")
# Try both SUPABASE_KEY (deployment) and SUPABASE_ANON_KEY (development)
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    # More informative error for debugging
    available_vars = [key for key in os.environ.keys() if 'SUPABASE' in key.upper()]
    raise ValueError(
        f"Missing Supabase configuration. "
        f"Required: SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY). "
        f"Available SUPABASE vars: {available_vars}"
    )

print(f"✅ Supabase configuration loaded: URL={SUPABASE_URL[:20]}..., KEY=***{SUPABASE_KEY[-4:] if SUPABASE_KEY else 'None'}")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_client() -> Client:
    """Get the Supabase client instance"""
    return supabase
