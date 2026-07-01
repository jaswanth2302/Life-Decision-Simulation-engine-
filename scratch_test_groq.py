import asyncio
import uuid
import httpx

async def test_groq_gemini_coexistence():
    agent_id = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        print("Initializing interview (Groq text generation)...")
        res = await client.post("http://localhost:8000/api/interview/initialize", json={"agent_id": agent_id})
        print(f"Init status: {res.status_code}")
        data = res.json()
        print(f"Question: {data['next_question']}")
        
        thread_id = data['thread_id']
        
        print("\nSubmitting response (Groq evaluation + Gemini embedding)...")
        res_submit = await client.post(
            "http://localhost:8000/api/interview/submit",
            json={
                "thread_id": thread_id,
                "agent_id": agent_id,
                "user_response": "I prefer low-risk investments and highly value academic education."
            }
        )
        print(f"Submit status: {res_submit.status_code}")
        submit_data = res_submit.json()
        print(f"Next question: {submit_data['next_question']}")
        print(f"Scores: {submit_data['evaluation_scores']}")

if __name__ == '__main__':
    asyncio.run(test_groq_gemini_coexistence())
