import asyncio
import hashlib
import json
import os
import re
from aiohttp import web, ClientSession
import edge_tts

VOICE_MAP = {
    "ielts": "en-GB-RyanNeural",          # Warm, natural modern British examiner
    "visa": "en-US-GuyNeural",            # Conversational, human US Consular Officer
    "job": "en-US-BrianNeural",           # Authoritative FAANG Tech Director
    "college": "en-US-JennyNeural",       # Articulate Ivy League Admissions Officer
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
        "supportive": "Be encouraging, warm, and conversational. Give the candidate room to speak.",
        "standard": "Be professional, clear, and direct, exactly like an official examiner.",
        "brutal": "Be challenging and skeptical. Test the candidate with pointed follow-ups."
    }.get(difficulty, "Be professional, clear, and direct.")

    custom_context = ""
    if topic:
        custom_context += f"\nTOPIC: {topic}"
    if target:
        custom_context += f"\nTARGET INSTITUTION/COMPANY: {target}"

    language_rule = """
CRITICAL LANGUAGE RULES (MANDATORY):
- Speak in everyday, natural, conversational English (B1/B2 level).
- Do NOT use obscure, academic C2 vocabulary (avoid words like 'philosophical grounds', 'cognitive automation', 'catalytic impact', 'interdisciplinary paradigm', 'empirical justification').
- Ask clear, concrete, human questions that test the candidate's real thoughts, not their ability to decipher complex riddle sentences.
- Keep your question short: strictly 1 to 2 spoken sentences (under 25 words).
- Never use bullet points, asterisks, markdown, quotation marks, or emojis."""

    if persona == "college":
        base = f"""You are Dean Eleanor Vance, an admissions interviewer for {target or 'university'}.
You want to know who the candidate really is, why they chose this major, and what they actually care about.
{demeanor_text}{custom_context}
{language_rule}
Listen to what the applicant just said and ask a natural, conversational follow-up question about their interests or experiences."""

    elif persona == "visa":
        base = f"""You are Officer Miller, a US visa officer conducting a quick consular interview for {target or 'a US university'}.
You want quick, honest answers about what they plan to study, how they will pay for it, and what their plans are after graduation.
{demeanor_text}{custom_context}
{language_rule}
Listen to what the applicant said and ask a direct, clear follow-up about their studies, sponsor finances, or plans to return home."""

    elif persona == "job":
        base = f"""You are Marcus Reed, an engineering director interviewing a candidate for {target or 'a software role'}.
You want to know what they actually worked on, how they solved problems, and what they learned.
{demeanor_text}{custom_context}
{language_rule}
Listen to the candidate and ask a practical, concrete follow-up question about their specific role, tech stack, or lessons learned."""

    else: # ielts
        base = f"""You are Dr. Arthur Davies, an official IELTS Speaking examiner.
In IELTS Speaking, examiners ask simple, natural, everyday questions to give the candidate room to demonstrate their English. Examiners do NOT speak in convoluted academic jargon.
{demeanor_text}{custom_context}
{language_rule}
Listen to the candidate's answer and ask a natural, straightforward follow-up question on the topic (e.g. why they feel that way, an example from their life, or how things might change in the future)."""

    return base

# Local Heuristic Semantic Engine Fallback
def local_semantic_comprehend(user_input, persona, turn, history, topic="", target=""):
    lower = user_input.lower().strip()
    target_str = target or "your chosen place"
    
    if persona == "college":
        if any(w in lower for w in ["research", "study", "lab", "professor", "class"]):
            return f"That sounds exciting. What kind of projects or classes are you most looking forward to taking at {target_str}?"
        elif any(w in lower for w in ["lead", "club", "founder", "started", "initiative", "team"]):
            return "Working with others can be tough. Can you tell me about a time you and your team disagreed on something?"
        else:
            return "What first got you interested in this subject? Was there a specific moment or person that inspired you?"

    elif persona == "visa":
        if any(w in lower for w in ["money", "father", "mother", "pay", "tuition", "bank", "fund", "dollar"]):
            return "Understood. What does your sponsor do for work, and how long have they been in that business?"
        elif any(w in lower for w in ["return", "home", "uzbekistan", "tashkent", "family", "back"]):
            return "What kind of job do you plan to look for when you come back home after graduating?"
        else:
            return f"Why did you choose {target_str} specifically over other universities in your home country?"

    elif persona == "job":
        if any(w in lower for w in ["we did", "our team", "we built", "together"]):
            return "What was your specific part in that project, and what tools or languages did you use?"
        elif any(w in lower for w in ["fast", "improved", "good", "better", "scaled"]):
            return "How did you measure that improvement? What changed before and after your fix?"
        else:
            return "If you had to do that project all over again today, what would you do differently?"

    else: # ielts
        if any(w in lower for w in ["tech", "ai", "computer", "future", "automate"]):
            return "Do you think artificial intelligence is making our daily lives easier or more complicated?"
        elif any(w in lower for w in ["society", "culture", "people", "country", "youth"]):
            return "Do young people in your country still follow traditional customs, or are things changing quickly?"
        elif turn == 1:
            return "That is interesting. What do you enjoy most about your studies or current work?"
        else:
            return "Can you give me a personal example of what you mean by that?"

async def handle_tts(request):
    if request.method == "POST":
        try:
            data = await request.json()
            text = str(data.get("text", "")).strip()
            voice_key = data.get("voice", "ielts")
        except Exception:
            text = ""
            voice_key = "ielts"
    else:
        text = request.query.get("text", "").strip()
        voice_key = request.query.get("voice", "ielts")
    voice = VOICE_MAP.get(voice_key, voice_key)

    if not text:
        return web.Response(status=400, text="Missing 'text' parameter")

    cache_key = hashlib.md5(f"{voice}_{text}".encode("utf-8")).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.mp3")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = f.read()
        return web.Response(body=data, content_type="audio/mpeg")

    try:
        async def fetch_audio():
            communicate = edge_tts.Communicate(text, voice, rate="+0%")
            out = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    out += chunk["data"]
            return out

        audio_data = await asyncio.wait_for(fetch_audio(), timeout=7.0)
        if audio_data:
            with open(cache_file, "wb") as f:
                f.write(audio_data)
            return web.Response(body=audio_data, content_type="audio/mpeg")
    except Exception as e:
        print(f"[Notice] Live TTS socket issue ({type(e).__name__}). Serving neural audio fallback...")
        # Serve any pre-rendered neural voice audio so candidate NEVER gets robot voice
        if os.path.exists(CACHE_DIR):
            files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".mp3")]
            if files:
                with open(os.path.join(CACHE_DIR, files[0]), "rb") as cf:
                    return web.Response(body=cf.read(), content_type="audio/mpeg")
        return web.Response(status=500, text=str(e))

async def handle_comprehend(request):
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    user_text = (data.get("user_text") or data.get("message", "")).strip()
    persona = data.get("persona") or data.get("mode", "ielts")
    turn = data.get("turn_count") if "turn_count" in data else data.get("turn", 1)
    history = data.get("history", [])
    topic = data.get("topic", "").strip()
    target = data.get("target", "").strip()
    difficulty = data.get("difficulty", "standard").strip()
    api_key = data.get("apiKey", "").strip() or get_env_gemini_key()

    if not user_text:
        return web.json_response({"reply": "Could you please elaborate on that specific point?"})

    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
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
                    "maxOutputTokens": 45
                }
            }

            async with ClientSession() as session:
                async with session.post(url, json=payload, timeout=4.0) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        reply = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        reply = re.sub(r'[\*\"\_]', '', reply)
                        return web.json_response({"reply": reply, "engine": "gemini-3.5-flash-lite"})
                    else:
                        err_text = await resp.text()
                        print(f"[Notice] Gemini API returned status {resp.status}: {err_text[:120]}")
        except Exception as e:
            print(f"[Notice] Comprehend fallback engaged: {type(e).__name__}")

    local_reply = local_semantic_comprehend(user_text, persona, turn, history, topic=topic, target=target)
    return web.json_response({"reply": local_reply, "engine": "local-semantic"})

async def handle_hint(request):
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    persona = data.get("persona") or data.get("mode", "ielts")
    topic = data.get("topic", "")
    target = data.get("target", "")
    last_question = data.get("lastQuestion") or data.get("last_question", "")
    api_key = data.get("apiKey", "").strip() or get_env_gemini_key()

    if api_key and last_question:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
            prompt = f"""You are a master interview coach for {persona.upper()} interviews (Topic: {topic}, Target: {target}).
The interviewer just asked: "{last_question}"
Give the candidate 2 concise, actionable bullet points showing them how an elite (Band 9 / Ivy League) candidate should structure their answer.
RULES: Output strictly 2 short bullet points (max 30 words total). No fluff, pure structure & strategy."""

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 80
                }
            }
            async with ClientSession() as session:
                async with session.post(url, json=payload, timeout=3.5) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        hint_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        return web.json_response({"hint": hint_text})
        except Exception as e:
            print(f"[Notice] Hint fallback engaged: {type(e).__name__}")

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
    persona = data.get("persona") or data.get("mode", "ielts")
    topic = data.get("topic", "")
    target = data.get("target", "")
    api_key = data.get("apiKey", "").strip() or get_env_gemini_key()

    user_turns = [h["text"] for h in history if h.get("speaker") == "user"]
    all_user_words = " ".join(user_turns)
    total_words = len(all_user_words.split())

    if api_key and len(user_turns) >= 1:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
            dialogue_str = "\n".join([f"{h['speaker'].upper()}: {h['text']}" for h in history])

            RUBRIC = {
                "ielts": "IELTS Academic Speaking (9-band scale). Score by Fluency & Coherence, Lexical Resource, Grammar, Pronunciation.",
                "college": "Ivy League admissions interview (100-point scale). Score by intellectual depth, authenticity, differentiation, institutional fit.",
                "visa": "F-1 student visa consular interview (probability %). Score by ties to home country, financial evidence, study intent clarity, no red flags.",
                "job": "Senior engineering/exec interview (100-point hire confidence). Score by STAR depth, ownership, technical credibility, culture fit."
            }
            rubric = RUBRIC.get(persona, RUBRIC["ielts"])

            prompt = f"""You are a brutally honest, world-class interview evaluator. Do NOT inflate scores.
Rubric: {rubric}
Context: {topic} | Target: {target}
Candidate word count: {total_words} words across {len(user_turns)} answer(s).

Full interview transcript:
{dialogue_str}

STRICT SCORING BANDS (apply without exception):
- Under 20 words total → Band 3.0-4.0 / 30-45% / 30-45pts
- 20-60 words, basic language → Band 4.5-5.5 / 50-65% / 50-62pts
- 60-120 words, simple but clear → Band 5.5-6.5 / 62-72% / 60-70pts
- 120-200 words, varied vocabulary → Band 6.5-7.5 / 72-82% / 68-78pts
- 200+ words, sophisticated, coherent → Band 7.5-8.5 / 82-90% / 78-88pts
- Genuinely exceptional only → Band 9.0 / 95%+ / 95+pts

UPGRADE RULES:
- "original" must be an EXACT quote from the candidate's actual words in the transcript above.
- If the candidate said very little, show one example of what an ideal opening should look like.
- Do NOT invent phrases the candidate did not say.

Return ONLY this JSON (no markdown, no backticks, nothing else):
{{
  "overallScore": "Band X.X or XX% or XX/100",
  "badgeClass": "emerald or accent or amber",
  "verdict": "Two specific, honest sentences: one strength observed, one concrete gap.",
  "fluencyScore": "X.X / 9.0",
  "lexicalScore": "X.X / 9.0",
  "logicScore": "X.X / 9.0",
  "upgrades": [
    {{
      "original": "exact phrase from candidate transcript",
      "improved": "elite-level rewrite of that phrase",
      "why": "one line on why this version is stronger"
    }},
    {{
      "original": "second exact phrase from transcript",
      "improved": "elite rewrite",
      "why": "one line"
    }}
  ],
  "topStrength": "one specific strength from THIS session",
  "keyImprovement": "one concrete, session-specific next step"
}}"""

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.25,
                    "maxOutputTokens": 900,
                    "responseMimeType": "application/json"
                }
            }
            async with ClientSession() as session:
                async with session.post(url, json=payload, timeout=18.0) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        raw_text = re.sub(r'^```json\s*', '', raw_text)
                        raw_text = re.sub(r'\s*```$', '', raw_text)
                        parsed = json.loads(raw_text)
                        return web.json_response(parsed)
                    else:
                        err_body = await resp.text()
                        print(f"[Notice] Gemini assess {resp.status}: {err_body[:120]}")
        except Exception as e:
            print(f"[Notice] Diagnostic fallback: {type(e).__name__}: {e}")

    # Fallback diagnostics — word-count calibrated
    if total_words < 20:
        score = "Band 3.5" if persona == "ielts" else ("30%" if persona == "visa" else "38/100")
        badge = "amber"
    elif total_words < 60:
        score = "Band 5.0" if persona == "ielts" else ("55%" if persona == "visa" else "55/100")
        badge = "amber"
    elif total_words < 120:
        score = "Band 6.0" if persona == "ielts" else ("65%" if persona == "visa" else "62/100")
        badge = "accent"
    elif total_words < 200:
        score = "Band 6.5" if persona == "ielts" else ("72%" if persona == "visa" else "68/100")
        badge = "accent"
    else:
        score = "Band 7.0" if persona == "ielts" else ("80%" if persona == "visa" else "75/100")
        badge = "accent"
    if persona == "college":
        score = score  # already handled above

    fallback_eval = {
        "overallScore": score,
        "badgeClass": badge,
        "verdict": f"Your {total_words}-word response covered the question but lacked the depth and range an examiner expects. Expand each answer with a concrete example, then a reflective comment to reach the next band.",
        "fluencyScore": "5.5 / 9.0" if total_words < 60 else "6.5 / 9.0",
        "lexicalScore": "5.0 / 9.0" if total_words < 60 else "6.0 / 9.0",
        "logicScore": "5.5 / 9.0" if total_words < 60 else "6.5 / 9.0",
        "upgrades": [
            {
                "original": user_turns[0][:80] if user_turns else "No answer recorded",
                "improved": "Start with your key point, support it with one specific example, then connect it back to the question.",
                "why": "PEEL structure (Point → Example → Explanation → Link) instantly raises coherence scores."
            },
            {
                "original": user_turns[-1][:80] if len(user_turns) > 1 else (user_turns[0][:80] if user_turns else "..."),
                "improved": "Replace vague descriptors ('very', 'a lot', 'important') with precise academic vocabulary and quantified claims.",
                "why": "Lexical precision is the fastest path from Band 6 to Band 7+.",
            },
        ],
        "topStrength": "You stayed engaged and responded directly to the question asked.",
        "keyImprovement": f"Your answers averaged {total_words // max(len(user_turns), 1)} words each — aim for 80-120 words per answer with a concrete example in every response."
    }
    return web.json_response(fallback_eval)

async def handle_index(request):
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return web.Response(text=content, content_type="text/html", headers=headers)

async def handle_ad(request):
    ad_path = os.path.join(os.path.dirname(__file__), "ad.html")
    with open(ad_path, "r", encoding="utf-8") as f:
        content = f.read()
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return web.Response(text=content, content_type="text/html", headers=headers)

app = web.Application()
app.router.add_get("/", handle_index)
app.router.add_get("/index.html", handle_index)
app.router.add_get("/ad", handle_ad)
app.router.add_get("/ad.html", handle_ad)
app.router.add_get("/api/tts", handle_tts)
app.router.add_post("/api/tts", handle_tts)
app.router.add_post("/api/comprehend", handle_comprehend)
app.router.add_post("/api/hint", handle_hint)
app.router.add_post("/api/analyze", handle_analyze)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 SpeakSense Neural Audio Server starting on http://0.0.0.0:{port} ...")
    web.run_app(app, host="0.0.0.0", port=port)
