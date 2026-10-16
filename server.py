import asyncio
import hashlib
import json
import os
import re
from aiohttp import web, ClientSession
import edge_tts

VOICE_MAP = {
    "ielts": "en-GB-RyanNeural",         # British Oxford Examiner
    "visa": "en-US-AndrewNeural",        # Authoritative, Conversational US Consular Officer
    "job": "en-US-BrianNeural",          # Sincere, Sharp FAANG Tech Director
    "default": "en-GB-RyanNeural"
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_env_gemini_key():
    for env_path in [os.path.expanduser("~/.env"), os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
    return os.environ.get("GEMINI_API_KEY", "").strip()

# System Prompts for Personas
SYSTEM_PROMPTS = {
  "ielts": """You are Dr. Arthur Davies, a strict senior British Council IELTS Speaking examiner.
The user is a candidate answering your questions.
Listen carefully to what the user said. Directly reference their ideas, point out any logical assumptions or interesting details, and ask a probing, high-band follow-up question.
CRITICAL RULES:
- Output strictly 1 to 2 spoken sentences (maximum 35 words).
- Do NOT use bullet points, asterisks, quotation marks, or emojis.
- Speak in formal, sophisticated British English.
- Be polite but academically rigorous.""",

  "visa": """You are Consular Officer Miller at the US Embassy interviewing an applicant for an F-1 student visa.
You are skeptical, direct, and under time pressure.
Address what the applicant just said. Question their funding, suspect immigrant intent (Section 214b), or demand specific proof of why they must return to Uzbekistan after graduation.
CRITICAL RULES:
- Output strictly 1 to 2 short, rapid-fire spoken sentences (maximum 30 words).
- Do NOT use bullet points, emojis, or markdown.
- Sound authoritative, professional, and slightly skeptical.""",

  "job": """You are Marcus Reed, a Senior Engineering Director at a top tech company conducting a behavioral interview.
The candidate is describing their experience.
Listen to their response and press them on the STAR framework (Situation, Task, Action, Result).
Ask them for specific metrics, their individual contribution (not the team's), or what they would do differently.
CRITICAL RULES:
- Output strictly 1 to 2 spoken sentences (maximum 35 words).
- Do NOT use emojis, bullet points, or markdown.
- Sound collaborative, sharp, and focused on measurable impact."""
}

# Local Heuristic Semantic Engine (When no external API key is provided)
def local_semantic_comprehend(user_input, persona, turn, history):
    text = user_input.strip()
    lower = text.lower()
    
    # Extract entities
    mentioned_uzbekistan = any(w in lower for w in ["tashkent", "uzbekistan", "samarkand", "bukhara", "home"])
    mentioned_tech = any(w in lower for w in ["code", "coding", "software", "tech", "computer", "ai", "algorithm", "python", "math"])
    mentioned_money = any(w in lower for w in ["money", "parent", "father", "mother", "business", "dollar", "afford", "tuition", "bank", "pay"])
    mentioned_future = any(w in lower for w in ["will", "future", "plan", "hope", "graduate", "career", "job", "company"])
    
    if persona == "ielts":
        if mentioned_tech:
            return "That raises a critical question about technological reliance. Do you believe that excessive automation will ultimately erode human creative thinking, or enhance it?"
        elif mentioned_uzbekistan:
            return "Rapid urbanization certainly brings both convenience and strain. What specific policy would you recommend local leaders implement to mitigate the resulting congestion?"
        elif any(w in lower for w in ["people", "society", "generation", "young"]):
            return "Social dynamics are shifting quickly. Do you think younger generations place too much emphasis on individual freedom over familial duty?"
        elif turn == 1:
            return "That provides useful context. Now, looking at your educational background, what specific experience first motivated you to pursue this trajectory?"
        else:
            return "An intriguing viewpoint. But if an opponent argued the exact opposite based on empirical evidence, how would you defend your claim?"

    elif persona == "visa":
        if mentioned_money:
            return "You mentioned family financing, but bank statements can be borrowed. Can you state your sponsor's monthly audited income and tax history?"
        elif mentioned_tech:
            return "Technology degrees are available worldwide. What specific research laboratory does this university offer that is unavailable in your home country?"
        elif mentioned_future:
            return "That sounds like a broad ambition. Do you have a binding employment pre-contract or physical property ties ensuring you will return to Tashkent?"
        else:
            return "I need a more concrete answer. What guarantee does the US Embassy have that you will not seek unlawful employment after graduation?"

    else: # job
        if any(w in lower for w in ["we did", "we worked", "our team"]):
            return "You mentioned what the team achieved, but what was your specific individual contribution to resolving that bottleneck?"
        elif any(w in lower for w in ["fast", "good", "better", "improved"]):
            return "Can you quantify that improvement with concrete metrics, such as percentage latency reduced or revenue gained?"
        else:
            return "Reflecting on that outcome, what was the biggest mistake you personally made during the implementation phase?"

async def handle_tts(request):
    text = request.query.get("text", "").strip()
    voice_key = request.query.get("voice", "ielts")
    voice = VOICE_MAP.get(voice_key, voice_key)

    if not text:
        return web.Response(status=400, text="Missing 'text' query parameter")

    cache_key = hashlib.md5(f"{voice}_{text}".encode("utf-8")).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.mp3")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = f.read()
        return web.Response(body=data, content_type="audio/mpeg")

    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        with open(cache_file, "wb") as f:
            f.write(audio_data)

        return web.Response(body=audio_data, content_type="audio/mpeg")
    except Exception as e:
        print(f"TTS Error: {e}")
        return web.Response(status=500, text=str(e))

async def handle_comprehend(request):
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    user_text = data.get("message", "").strip()
    persona = data.get("persona", "ielts")
    turn = data.get("turn", 1)
    history = data.get("history", [])
    api_key = data.get("apiKey", "").strip() or get_env_gemini_key()

    if not user_text:
        return web.json_response({"reply": "Could you please elaborate on that point?"})

    # If Gemini API Key is provided, use Google's cutting edge neural reasoning!
    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
            
            # Format history for Gemini
            contents = []
            sys_inst = SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["ielts"])
            
            for h in history[-6:]: # Keep last 6 turns
                role = "user" if h.get("speaker") == "user" else "model"
                contents.append({"role": role, "parts": [{"text": h.get("text", "")}]})
            
            # Add latest user input
            contents.append({"role": "user", "parts": [{"text": user_text}]})

            payload = {
                "systemInstruction": {"parts": [{"text": sys_inst}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 60,
                    "topP": 0.95
                }
            }

            async with ClientSession() as session:
                async with session.post(url, json=payload, timeout=8) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        reply = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        # Clean reply from quotes or markdown
                        reply = re.sub(r'[\*\"\_]', '', reply)
                        return web.json_response({"reply": reply, "engine": "gemini-3.8-flash"})
                    else:
                        err_text = await resp.text()
                        print(f"Gemini API returned status {resp.status}: {err_text}")
        except Exception as e:
            print(f"Gemini request failed: {e}")

    # Fallback to local high-precision semantic comprehension
    local_reply = local_semantic_comprehend(user_text, persona, turn, history)
    return web.json_response({"reply": local_reply, "engine": "local-semantic"})

async def handle_index(request):
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return web.Response(text=content, content_type="text/html")

app = web.Application()
app.router.add_get("/", handle_index)
app.router.add_get("/index.html", handle_index)
app.router.add_get("/api/tts", handle_tts)
app.router.add_post("/api/comprehend", handle_comprehend)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 SpeakSense Neural Audio Server starting on http://0.0.0.0:{port} ...")
    web.run_app(app, host="0.0.0.0", port=port)
