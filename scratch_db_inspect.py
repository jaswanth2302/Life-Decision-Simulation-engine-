import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from supabase import acreate_client

async def inspect():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    client = await acreate_client(url, key)
    
    # Try inserting a dummy agent to see what fields are required or if it succeeds
    import uuid
    agent_id = str(uuid.uuid4())
    try:
        res = await client.table("agents").insert({"id": agent_id}).execute()
        print("Insert simple id success:", res.data)
        # Clean up
        await client.table("agents").delete().eq("id", agent_id).execute()
    except Exception as e:
        print("Insert simple id failed:", e)

if __name__ == "__main__":
    asyncio.run(inspect())
