import asyncio
import hashlib
import json
import os
import re
from aiohttp import web, ClientSession
import edge_tts

VOICE_MAP = {
    "ielts": "en-GB-RyanNeural",          # British Oxford Examiner
    "visa": "en-US-AndrewNeural",         # Authoritative, Conversational US Consular Officer
    "job": "en-US-BrianNeural",           # Sincere, Sharp FAANG Tech Director
    "college": "en-US-AvaNeural",         # Intellectually Sharp Ivy League Admissions Officer
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

def build_system_prompt(persona, topic="", target="", difficulty="standard"):
    demeanor_text = {
        "supportive": "Your demeanor is warm, encouraging, and collaborative. Give the candidate constructive room to shine, but still ask thought-provoking questions.",
        "standard": "Your demeanor is formal, realistic, and objective, adhering strictly to official interview standards.",
        "brutal": "Your demeanor is relentless, skeptical, and high-pressure. Play devil's advocate, expose logical fallacies, cross-examine suspicious claims, and challenge every assumption."
    }.get(difficulty, "Your demeanor is realistic and objective.")

    custom_context = ""
    if topic:
        custom_context += f"\nINTERVIEW FOCUS TOPIC: {topic}"
    if target:
        custom_context += f"\nTARGET INSTITUTION/ORGANIZATION: {target}"

    if persona == "college":
        base = f"""You are Dean Eleanor Vance, Senior Admissions Officer evaluating an applicant for an elite university.
The candidate is answering your questions about their background, intellectual vitality, leadership, and reasons for applying.
{demeanor_text}{custom_context}
Listen closely to what the candidate said. Reference their specific words or ideas, probe their deeper motivation, and test if they have genuine intellectual curiosity rather than rehearsed answers.
CRITICAL RULES:
- Output strictly 1 to 2 spoken sentences (maximum 35 words).
- Do NOT use bullet points, asterisks, quotation marks, or emojis.
- Sound engaging, intellectually rigorous, and authentic."""

    elif persona == "visa":
        base = f"""You are Consular Officer Miller at the US Embassy conducting an F-1/B-1 visa interview.
The applicant is standing in front of you. You are under tight time constraints and skeptical by nature.
{demeanor_text}{custom_context}
Listen to what the applicant just said. Zero in on funding sources, family income plausibility, specific institutional choice, or ties to their home country (Section 214b).
CRITICAL RULES:
- Output strictly 1 to 2 rapid-fire spoken sentences (maximum 30 words).
- Do NOT use bullet points, markdown, or emojis.
- Sound authoritative, professional, and slightly skeptical."""

    elif persona == "job":
        base = f"""You are Marcus Reed, Senior Engineering Director at a top technology company conducting a behavioral and technical depth interview.
{demeanor_text}{custom_context}
Listen to what the candidate said and evaluate them using the STAR framework (Situation, Task, Action, Result).
Press them on their individual ownership, trade-offs, engineering bottlenecks, or what they would do differently.
CRITICAL RULES:
- Output strictly 1 to 2 spoken sentences (maximum 35 words).
- Do NOT use markdown, emojis, or bullet points.
- Sound sharp, collaborative, and focused on measurable impact."""

    else: # ielts
        base = f"""You are Dr. Arthur Davies, a senior British Council IELTS Speaking examiner.
The candidate is taking the IELTS Speaking test.
{demeanor_text}{custom_context}
Listen carefully to the candidate's response. Directly reference their ideas, point out logical nuances, and ask a probing follow-up question that tests complex vocabulary and abstract reasoning (Band 7 to 9 level).
CRITICAL RULES:
- Output strictly 1 to 2 spoken sentences (maximum 35 words).
- Do NOT use bullet points, asterisks, quotation marks, or emojis.
- Speak in formal, sophisticated British English."""

    return base

# Local Heuristic Semantic Engine Fallback
def local_semantic_comprehend(user_input, persona, turn, history, topic="", target=""):
    lower = user_input.lower().strip()
    target_str = target or "your target institution"
    
    if persona == "college":
        if any(w in lower for w in ["research", "study", "lab", "professor", "class"]):
            return f"That specific academic focus is compelling. How do you plan to leverage {target_str} unique interdisciplinary labs to contribute beyond your coursework?"
        elif any(w in lower for w in ["lead", "club", "founder", "started", "initiative"]):
            return "Leading an initiative often uncovers unexpected resistance. What was the sharpest ethical or logistical trade-off you had to navigate during that project?"
        else:
            return f"Every competitive applicant mentions passion, but sustained curiosity requires resilience. When was the last time you pursued a problem where you initially failed repeatedly?"

    elif persona == "visa":
        if any(w in lower for w in ["money", "father", "mother", "pay", "tuition", "bank", "fund", "dollar"]):
            return "Bank balances alone do not prove stability. Can you state your sponsor's monthly net tax-audited revenue and business stability over recent years?"
        elif any(w in lower for w in ["return", "home", "uzbekistan", "tashkent", "family"]):
            return "Many students state intent to return, yet seek OPT and stay. What specific contractual commitment ties you to returning immediately upon graduation?"
        else:
            return f"Degrees in this discipline are widely accessible domestically. What exact specialization does {target_str} offer that justifies foreign expenditure?"

    elif persona == "job":
        if any(w in lower for w in ["we did", "our team", "we built", "together"]):
            return "You mentioned the group effort, but what was your individual architectural contribution when resolving the critical bottleneck?"
        elif any(w in lower for w in ["fast", "improved", "good", "better", "scaled"]):
            return "Can you quantify that outcome with concrete engineering metrics, such as latency percentiles, error rates, or infrastructure costs saved?"
        else:
            return "Looking back on that technical decision, what was the biggest hidden downside or technical debt that resulted from your approach?"

    else: # ielts
        if any(w in lower for w in ["tech", "ai", "computer", "future", "automate"]):
            return "That highlights an ongoing philosophical debate. Do you believe cognitive automation will ultimately diminish human creativity, or democratize it?"
        elif any(w in lower for w in ["society", "culture", "people", "country", "youth"]):
            return "Traditional values often clash with rapid modernization. Should policymakers actively preserve cultural heritage at the expense of economic velocity?"
        elif turn == 1:
            return "That provides helpful initial context. Looking at your academic foundation, what pivotal moment confirmed this was your dedicated path?"
        else:
            return "That is an intriguing proposition. However, if empirical evidence suggested the exact contrary, on what philosophical grounds would you defend your assertion?"

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
        communicate = edge_tts.Communicate(text, voice, rate="+2%")
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
    topic = data.get("topic", "").strip()
    target = data.get("target", "").strip()
    difficulty = data.get("difficulty", "standard").strip()
    api_key = data.get("apiKey", "").strip() or get_env_gemini_key()

    if not user_text:
        return web.json_response({"reply": "Could you please elaborate on that specific point?"})

    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
            sys_inst = build_system_prompt(persona, topic=topic, target=target, difficulty=difficulty)

            contents = []
            for h in history[-6:]:
                role = "user" if h.get("speaker") == "user" else "model"
                contents.append({"role": role, "parts": [{"text": h.get("text", "")}]})
            
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
                        reply = re.sub(r'[\*\"\_]', '', reply)
                        return web.json_response({"reply": reply, "engine": "gemini-3.8-flash"})
                    else:
                        err_text = await resp.text()
                        print(f"Gemini API returned status {resp.status}: {err_text}")
        except Exception as e:
            print(f"Gemini request failed: {e}")

    local_reply = local_semantic_comprehend(user_text, persona, turn, history, topic=topic, target=target)
    return web.json_response({"reply": local_reply, "engine": "local-semantic"})

async def handle_hint(request):
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    persona = data.get("persona", "ielts")
    topic = data.get("topic", "")
    target = data.get("target", "")
    last_question = data.get("lastQuestion", "")
    api_key = data.get("apiKey", "").strip() or get_env_gemini_key()

    if api_key and last_question:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
            prompt = f"""You are a master interview coach for {persona.upper()} interviews (Topic: {topic}, Target: {target}).
The interviewer just asked: "{last_question}"
Give the candidate 2 concise, actionable bullet points showing them how an elite (Band 9 / Ivy League) candidate should structure their answer.
RULES: Output strictly 2 short bullet points (max 30 words total). No fluff, pure structure & strategy."""

            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.5, "maxOutputTokens": 80}
            }
            async with ClientSession() as session:
                async with session.post(url, json=payload, timeout=6) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        hint_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        return web.json_response({"hint": hint_text})
        except Exception as e:
            print(f"Hint error: {e}")

    # Fallback hints based on persona
    fallback_map = {
        "college": "• Acknowledge the core challenge, then cite a concrete project where you took intellectual initiative.\n• Connect your personal trajectory directly to a distinctive resource at this university.",
        "visa": "• State your sponsor's audited monthly income clearly in USD with zero hesitation.\n• Mention a specific job offer or property ties requiring your return to your home country.",
        "job": "• Use STAR: State the high-stakes Situation, your exact Action, and measurable metric outcome.\n• Acknowledge what didn't work and the key engineering lesson you adopted.",
        "ielts": "• Open with a sophisticated stance using academic markers (e.g. 'While it is ostensibly argued...').\n• Provide a balanced counterpoint before defending your conclusion with a tangible example."
    }
    return web.json_response({"hint": fallback_map.get(persona, fallback_map["ielts"])})

async def handle_analyze(request):
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    history = data.get("history", [])
    persona = data.get("persona", "ielts")
    topic = data.get("topic", "")
    target = data.get("target", "")
    api_key = data.get("apiKey", "").strip() or get_env_gemini_key()

    user_turns = [h["text"] for h in history if h.get("speaker") == "user"]
    all_user_words = " ".join(user_turns)
    total_words = len(all_user_words.split())

    if api_key and len(user_turns) >= 2:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
            dialogue_str = "\n".join([f"{h['speaker'].upper()}: {h['text']}" for h in history])
            
            prompt = f"""You are an elite interview auditor analyzing a candidate's spoken performance for {persona.upper()} (Topic: {topic}, Target: {target}).
Analyze this full conversation transcript:
{dialogue_str}

Return a valid JSON object matching this schema:
{{
  "overallScore": "string (e.g. 'Band 8.0' or 'Strong Admit 89/100' or 'Visa Approved 92%')",
  "badgeClass": "string ('emerald' for high, 'accent' for good, 'amber' for mediocre)",
  "verdict": "string (2 crisp analytical sentences summarizing overall fluency, rigor, and poise)",
  "fluencyScore": "string (e.g. '8.5 / 9.0')",
  "lexicalScore": "string (e.g. '8.0 / 9.0')",
  "logicScore": "string (e.g. '7.5 / 9.0')",
  "upgrades": [
    {{
      "original": "short quote of candidate's actual basic phrase",
      "improved": "high-impact Band 9 / Ivy League rewrite",
      "why": "brief explanation of why this elevates the argument"
    }},
    {{
      "original": "second phrase candidate used",
      "improved": "elevated academic rewrite",
      "why": "brief explanation"
    }}
  ],
  "topStrength": "one sentence highlight of candidate's best quality",
  "keyImprovement": "one actionable adjustment for next interview"
}}
CRITICAL: Output ONLY raw valid JSON. No markdown code blocks, no backticks."""

            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 600}
            }
            async with ClientSession() as session:
                async with session.post(url, json=payload, timeout=12) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        raw_text = re.sub(r'^```json\s*', '', raw_text)
                        raw_text = re.sub(r'\s*```$', '', raw_text)
                        parsed = json.loads(raw_text)
                        return web.json_response(parsed)
        except Exception as e:
            print(f"Analyze error: {e}")

    # Fallback diagnostics
    score = "Band 7.5" if total_words > 120 else "Band 6.5"
    if persona == "visa":
        score = "Visa Approved 88%" if total_words > 80 else "Review Needed (214b)"
    elif persona == "college":
        score = "Strong Contender 85/100" if total_words > 100 else "Needs Differentiation 72/100"

    fallback_eval = {
        "overallScore": score,
        "badgeClass": "emerald" if "Approved" in score or "8" in score else "accent",
        "verdict": "You maintained consistent dialogue velocity and addressed the interviewer's direct inquiries. To reach the top percentile, eliminate conversational fillers and replace colloquial verbs with domain-specific academic terminology.",
        "fluencyScore": "7.5 / 9.0",
        "lexicalScore": "7.0 / 9.0",
        "logicScore": "8.0 / 9.0",
        "upgrades": [
            {
                "original": "I want to study this because it is very important for the future",
                "improved": "I am driven to master this domain given its catalytic impact on next-generation systems",
                "why": "Replaces generic descriptors with purposeful academic articulation."
            },
            {
                "original": "I worked with my team to fix the big problem",
                "improved": "I orchestrated cross-functional efforts to remediate a critical architectural bottleneck",
                "why": "Demonstrates decisive individual leadership using active operational verbs."
            }
        ],
        "topStrength": "Responsive conversational adaptability under rapid-fire questioning.",
        "keyImprovement": "Lead immediately with concrete metrics rather than qualitative generalizations."
    }
    return web.json_response(fallback_eval)

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
app.router.add_post("/api/hint", handle_hint)
app.router.add_post("/api/analyze", handle_analyze)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 SpeakSense Neural Audio Server starting on http://0.0.0.0:{port} ...")
    web.run_app(app, host="0.0.0.0", port=port)
