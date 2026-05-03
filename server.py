"""
server.py — Flask backend for MediBot (MCP + Twilio Emergency edition).

Endpoints:
  GET  /                              → serve index.html
  GET  /api/health                    → key + store + Twilio status
  GET  /api/docs                      → list loaded documents & chunk count
  POST /api/upload-doc                → upload PDF/TXT into MCP store
  POST /api/analyze                   → audio + image + text → JSON response
  GET  /api/audio/<name>              → serve generated TTS audio

  POST /api/emergency/trigger         → place Twilio call + SMS to emergency contact
  GET  /api/emergency/twiml           → TwiML webhook (Twilio fetches this during the call)
  POST /api/emergency/status          → Twilio call-status callback (logs progress)
  GET  /api/emergency/call-status/<sid> → poll live call status from frontend
  GET  /api/emergency/config          → check Twilio env var status
"""

from __future__ import annotations
import os
import uuid
import json
import logging
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── local modules ────────────────────────────────────────────────────────────
import mcp_server as mcp
import emergency_call as ec
from brain_of_the_doctor import encode_image
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="static")

# ── directories ──────────────────────────────────────────────────────────────
DOCS_DIR   = Path("docs")
AUDIO_DIR  = Path("audio_out")
UPLOAD_DIR = Path("uploads")
for d in (DOCS_DIR, AUDIO_DIR, UPLOAD_DIR):
    d.mkdir(exist_ok=True)

# ── Groq client ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are acting as a professional doctor for educational purposes.
Analyze the patient's query and any image provided.

You have access to the following tools — use them proactively to give evidence-based answers:
  - search_pubmed: Use for ANY general clinical question, symptom, or disease. This searches 35M+ peer-reviewed articles.
  - lookup_drug: Use whenever a patient mentions a medication name or asks about dosing, interactions, or side effects.
  - lookup_condition: Use to find which drugs are associated with a reported symptom or condition.
  - search_local_knowledge: Use when the patient has uploaded institution-specific documents or personal records.

Tool usage rules:
  1. For general medical questions → always call search_pubmed first.
  2. For drug-related questions → always call lookup_drug.
  3. You may call multiple tools in sequence to give a complete answer.
  4. Do NOT skip tools to save time — the patient deserves evidence-backed responses.

Response rules:
  - If you make a differential diagnosis, suggest remedies.
  - Do not use numbers, bullet points, or special characters in your final response.
  - Respond in one concise paragraph (2–3 sentences max).
  - Always address the patient directly.
  - Begin with 'With what I see, I think...' when an image is provided.
  - Do not identify yourself as an AI. Write like a real doctor.
  - No preamble."""

SEVERITY_PROMPT = """You are a medical triage classifier. Given a patient's symptom description and/or a doctor's response, assess the severity of the condition.

Respond ONLY with a valid JSON object — no explanation, no markdown, no preamble:
{
  "severity": "critical" | "moderate" | "mild",
  "reason": "one sentence reason",
  "keywords": ["list", "of", "key", "symptoms"],
  "specialist": "dermatologist" | "cardiologist" | "neurologist" | "orthopedic" | "general physician" | "emergency"
}

Rules:
- "critical": life-threatening conditions requiring immediate emergency response.
  Examples: heart attack, stroke, severe chest pain, difficulty breathing, unconsciousness,
  severe allergic reaction (anaphylaxis), uncontrolled bleeding, poisoning, overdose,
  seizure, suspected spinal injury, severe burns, sepsis signs.
- "moderate": serious conditions needing urgent medical attention within hours but not immediately life-threatening.
  Examples: high fever, moderate pain, suspected fracture, persistent vomiting, infected wounds.
- "mild": non-urgent conditions manageable with home care or a routine doctor visit.
  Examples: common cold, minor cuts, mild headache, minor rash, low-grade fever.

For specialist field, pick the most relevant specialist based on symptoms:
- Skin rash, lesion, wound, burn, acne, eczema → dermatologist
- Chest pain, heart palpitations, shortness of breath → cardiologist
- Headache, seizure, numbness, dizziness → neurologist
- Bone, joint, muscle pain, fracture → orthopedic
- Life-threatening emergency → emergency
- Anything else → general physician"""


def classify_severity(user_query: str, doctor_response: str) -> dict:
    """
    Use LLM to classify whether the condition is critical/moderate/mild.
    Returns dict with keys: severity, reason, keywords
    """
    if not GROQ_API_KEY:
        return {"severity": "mild", "reason": "Could not classify — no API key.", "keywords": []}

    client = Groq(api_key=GROQ_API_KEY)
    combined = f"Patient query: {user_query}\n\nDoctor assessment: {doctor_response}"

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SEVERITY_PROMPT},
                {"role": "user",   "content": combined},
            ],
            max_tokens=200,
            temperature=0.1,   # low temperature for consistent classification
        )
        raw = resp.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        # Validate shape
        if result.get("severity") not in ("critical", "moderate", "mild"):
            result["severity"] = "mild"
        return result
    except Exception as e:
        logger.warning(f"Severity classification failed: {e}")
        return {"severity": "mild", "reason": "Classification error.", "keywords": []}


# ---------------------------------------------------------------------------
# Agentic tool-calling loop
# ---------------------------------------------------------------------------

def agentic_complete(
    user_content: list[dict],
    max_iterations: int = 5,
) -> tuple[str, list[dict]]:
    """
    Run the Groq tool-calling loop with MCP tools.

    Returns:
        (final_text_response, tool_calls_log)
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    tool_calls_log: list[dict] = []

    for iteration in range(max_iterations):
        logger.info(f"MCP loop iteration {iteration + 1}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=mcp.MCP_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # Append assistant turn to history
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (assistant_msg.tool_calls or [])
            ] or None,
        })

        # If no tool calls → final answer
        if not assistant_msg.tool_calls:
            return (assistant_msg.content or ""), tool_calls_log

        # Execute each tool call and feed results back
        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info(f"  Tool call: {fn_name}({fn_args})")
            result_str = mcp.dispatch_tool(fn_name, fn_args)
            tool_calls_log.append({"tool": fn_name, "args": fn_args, "result_preview": result_str[:200]})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Fallback: last assistant content (shouldn't normally reach here)
    last = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant" and m.get("content")),
        "I'm sorry, I was unable to generate a response.",
    )
    return last, tool_calls_log


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    groq_key = os.environ.get("GROQ_API_KEY", "")
    el_key   = os.environ.get("ELEVENLABS_API_KEY", "")
    summary  = mcp.store_summary()
    external_tools = ["search_pubmed", "lookup_drug", "lookup_condition"]
    local_tools = [t["function"]["name"] for t in mcp.MCP_TOOLS if t["function"]["name"] not in external_tools]
    return jsonify({
        "GROQ_API_KEY":       "set" if groq_key else "MISSING",
        "GROQ_KEY_PREFIX":    groq_key[:8] + "..." if groq_key else "n/a",
        "ELEVENLABS_API_KEY": "set" if el_key else "not set (gTTS fallback active)",
        "mcp_local_tools":    local_tools,
        "mcp_external_tools": external_tools,
        "external_apis":      summary.get("external_apis", []),
        "local_store":        {"documents": summary["documents"], "total_chunks": summary["total_chunks"]},
        "twilio":             ec.config_status(),
    })


@app.route("/api/docs", methods=["GET"])
def list_docs():
    return jsonify(mcp.store_summary())


@app.route("/api/upload-doc", methods=["POST"])
def upload_doc():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    dest = DOCS_DIR / f.filename
    f.save(str(dest))

    try:
        result = mcp.add_document(str(dest), label=f.filename)
        return jsonify({"message": f"Ingested '{f.filename}' → {result['chunks_added']} chunks", **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    audio_file = request.files.get("audio")
    image_file = request.files.get("image")

    # 1. Speech → Text
    speech_text = ""
    if audio_file and audio_file.filename:
        audio_path = UPLOAD_DIR / f"audio_{uuid.uuid4().hex}.webm"
        audio_file.save(str(audio_path))
        try:
            speech_text = transcribe_with_groq(
                stt_model="whisper-large-v3",
                audio_filepath=str(audio_path),
                GROQ_API_KEY=GROQ_API_KEY,
            )
        except RuntimeError as e:
            return jsonify({"error": f"Transcription failed: {e}"}), 500
        finally:
            audio_path.unlink(missing_ok=True)

    typed_text        = request.form.get("text", "").strip()
    emergency_contact = request.form.get("emergency_contact", "").strip()  # ← NEW
    patient_name      = request.form.get("patient_name", "the patient").strip()
    user_query        = (speech_text + " " + typed_text).strip() or "Please analyze my image."

    # 2. Build user content (text + optional image for multimodal)
    user_content: list[dict] = [{"type": "text", "text": user_query}]

    image_path = None
    if image_file and image_file.filename:
        image_path = UPLOAD_DIR / f"img_{uuid.uuid4().hex}.jpg"
        image_file.save(str(image_path))
        try:
            import base64
            with open(image_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        except Exception as e:
            logger.warning(f"Image encoding failed: {e}")

    # 3. Agentic MCP loop — LLM decides which tools to call
    try:
        doctor_response, tool_log = agentic_complete(user_content)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if image_path and image_path.exists():
            image_path.unlink(missing_ok=True)

    # 4. Severity classification ← NEW
    severity_result = classify_severity(user_query, doctor_response)
    severity        = severity_result.get("severity", "mild")
    logger.info(f"Severity: {severity} — {severity_result.get('reason', '')}")

    # 5. Auto emergency call if CRITICAL and contact number provided ← NEW
    emergency_result = None
    auto_called      = False
    if severity == "critical" and emergency_contact:
        logger.warning(f"CRITICAL condition detected — auto-calling {emergency_contact}")
        summary_for_call = (
            f"Patient ({patient_name}) reported: {user_query[:200]}. "
            f"Doctor assessment: {doctor_response[:250]}."
        )
        emergency_result = ec.trigger_emergency(
            summary      = summary_for_call,
            patient_name = patient_name,
            to_number    = emergency_contact,
        )
        auto_called = emergency_result.get("success", False)
        logger.info(f"Auto-call result: {emergency_result}")

    # 6. TTS
    audio_name = f"response_{uuid.uuid4().hex}.mp3"
    audio_out  = AUDIO_DIR / audio_name
    audio_url  = None
    try:
        if os.environ.get("ELEVENLABS_API_KEY"):
            text_to_speech_with_elevenlabs(
                input_text=doctor_response,
                output_filepath=str(audio_out),
            )
        else:
            text_to_speech_with_gtts(
                input_text=doctor_response,
                output_filepath=str(audio_out),
            )
        audio_url = f"/api/audio/{audio_name}"
    except Exception as e:
        logger.warning(f"TTS failed: {e}")

    return jsonify({
        "speech_text":       speech_text,
        "doctor_response":   doctor_response,
        "audio_url":         audio_url,
        "mcp_tools_used":    [t["tool"] for t in tool_log],
        "tool_call_count":   len(tool_log),
        # severity fields
        "severity":          severity,
        "severity_reason":   severity_result.get("reason", ""),
        "severity_keywords": severity_result.get("keywords", []),
        "specialist":        severity_result.get("specialist", "general physician"),
        "auto_called":       auto_called,
        "emergency_result":  emergency_result,
    })


@app.route("/api/nearby-doctors", methods=["POST"])
def nearby_doctors():
    """
    Find nearby specialists using OpenStreetMap Overpass API.
    100% free — no API key, no billing account required.

    JSON body:
      lat        float   patient latitude
      lng        float   patient longitude
      specialty  str     e.g. "dermatologist"
      radius     int     search radius in metres (default 5000)
    """
    import urllib.request, urllib.parse, math

    data      = request.get_json(silent=True) or {}
    lat       = data.get("lat")
    lng       = data.get("lng")
    specialty = data.get("specialty", "dermatologist")
    radius    = int(data.get("radius", 5000))

    if lat is None or lng is None:
        return jsonify({"error": "lat and lng are required"}), 400

    # ── Map specialty → OSM tags ──────────────────────────────────────────────
    SPECIALTY_TAGS = {
        "dermatologist":     [('healthcare:speciality', 'dermatology'),
                              ('amenity', 'doctors'),
                              ('amenity', 'clinic')],
        "cardiologist":      [('healthcare:speciality', 'cardiology'),
                              ('amenity', 'hospital'),
                              ('amenity', 'clinic')],
        "neurologist":       [('healthcare:speciality', 'neurology'),
                              ('amenity', 'hospital'),
                              ('amenity', 'clinic')],
        "orthopedic surgeon":[('healthcare:speciality', 'orthopaedics'),
                              ('amenity', 'hospital'),
                              ('amenity', 'clinic')],
        "general physician": [('amenity', 'doctors'),
                              ('amenity', 'clinic'),
                              ('healthcare', 'doctor')],
        "hospital emergency":[('amenity', 'hospital'),
                              ('emergency', 'yes')],
    }

    tags = SPECIALTY_TAGS.get(specialty, [('amenity', 'clinic'), ('amenity', 'doctors')])

    # Build Overpass QL query — union of all relevant tags
    union_parts = []
    for key, val in tags:
        union_parts.append(f'node["{key}"="{val}"](around:{radius},{lat},{lng});')
        union_parts.append(f'way["{key}"="{val}"](around:{radius},{lat},{lng});')

    overpass_query = f"""
[out:json][timeout:15];
(
  {''.join(union_parts)}
);
out center tags 20;
"""

    def haversine(lat1, lng1, lat2, lng2):
        """Distance in km between two GPS points."""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    try:
        payload = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=payload,
            headers={
                "User-Agent": "MediBot/2.0 (educational medical assistant; contact@example.com)",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            osm_data = json.loads(r.read().decode("utf-8"))

        elements = osm_data.get("elements", [])

        results = []
        seen_names = set()

        for el in elements:
            tags_el  = el.get("tags", {})
            name     = tags_el.get("name", "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            # Coordinates — nodes have lat/lng directly; ways have center
            if el.get("type") == "node":
                elat, elng = el.get("lat"), el.get("lon")
            else:
                center = el.get("center", {})
                elat, elng = center.get("lat"), center.get("lon")

            if not elat or not elng:
                continue

            dist_km   = haversine(lat, lng, elat, elng)
            address   = ", ".join(filter(None, [
                tags_el.get("addr:housenumber", ""),
                tags_el.get("addr:street", ""),
                tags_el.get("addr:suburb", ""),
                tags_el.get("addr:city", ""),
            ])) or tags_el.get("addr:full", "Address not listed")

            phone     = tags_el.get("phone", "") or tags_el.get("contact:phone", "")
            website   = tags_el.get("website", "") or tags_el.get("contact:website", "")
            opening   = tags_el.get("opening_hours", "")
            amenity   = tags_el.get("amenity", tags_el.get("healthcare", ""))

            results.append({
                "name":           name,
                "address":        address,
                "distance_km":    round(dist_km, 2),
                "phone":          phone,
                "website":        website,
                "opening_hours":  opening,
                "type":           amenity,
                "lat":            elat,
                "lng":            elng,
                "osm_id":         el.get("id"),
                "directions_url": f"https://www.google.com/maps/dir/{lat},{lng}/{elat},{elng}",
                "maps_url":       f"https://www.openstreetmap.org/?mlat={elat}&mlon={elng}&zoom=17",
            })

        # Sort by distance
        results.sort(key=lambda x: x["distance_km"])
        results = results[:8]

        return jsonify({
            "results":   results,
            "specialty": specialty,
            "radius_km": radius // 1000,
            "center":    {"lat": lat, "lng": lng},
            "source":    "OpenStreetMap (Overpass API) — free, no API key",
        })

    except Exception as e:
        logger.error(f"Overpass API error: {e}")
        return jsonify({"error": f"Map search failed: {str(e)}"}), 500


@app.route("/api/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(str(AUDIO_DIR), filename)


# ---------------------------------------------------------------------------
# Emergency calling — Twilio
# ---------------------------------------------------------------------------

@app.route("/api/emergency/trigger", methods=["POST"])
def emergency_trigger():
    """
    Trigger an emergency call + SMS.

    JSON body (all optional):
      summary       str   override the auto-generated summary
      patient_name  str   name to include in the alert (default "the patient")
      to_number     str   override EMERGENCY_TO_NUMBER for this call only
    """
    data         = request.get_json(silent=True) or {}
    summary      = data.get("summary", "The patient is experiencing a medical emergency and needs immediate assistance.")
    patient_name = data.get("patient_name", "the patient")
    to_number    = data.get("to_number") or None  # None → use env default

    result = ec.trigger_emergency(
        summary=summary,
        patient_name=patient_name,
        to_number=to_number,
    )

    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


@app.route("/api/emergency/twiml", methods=["GET", "POST"])
def emergency_twiml():
    """
    TwiML webhook — Twilio fetches this URL when the recipient answers.
    Summary and patient name are passed as query params by trigger_emergency().
    """
    from flask import Response as FlaskResponse
    summary      = request.args.get("summary", "Please respond immediately to a medical emergency.")
    patient_name = request.args.get("name", "the patient")
    xml = ec.build_twiml(summary, patient_name)
    return FlaskResponse(xml, mimetype="text/xml")


@app.route("/api/emergency/status", methods=["POST"])
def emergency_status_callback():
    """
    Twilio POSTs call progress events here (ringing, answered, completed, etc.).
    Logs them and returns 204.
    """
    call_sid    = request.form.get("CallSid", "unknown")
    call_status = request.form.get("CallStatus", "unknown")
    call_to     = request.form.get("To", "")
    logger.info(f"[Twilio] Call {call_sid} → {call_to} : {call_status}")
    return ("", 204)


@app.route("/api/emergency/call-status/<call_sid>", methods=["GET"])
def emergency_call_status(call_sid: str):
    """Poll the live status of a specific call SID."""
    return jsonify(ec.get_call_status(call_sid))


@app.route("/api/emergency/config", methods=["GET"])
def emergency_config():
    """Return Twilio configuration status (no secret values exposed)."""
    return jsonify(ec.config_status())


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    static_dir = Path("static")
    target = static_dir / path if path else static_dir / "index.html"
    if target.exists():
        return send_file(str(target))
    return send_file(str(static_dir / "index.html"))


if __name__ == "__main__":
    print("\n🩺  MediBot MCP + Twilio server starting...")
    print(f"   GROQ_API_KEY       : {'✅ set' if GROQ_API_KEY else '❌ missing'}")
    print(f"   ELEVENLABS_API_KEY : {'✅ set' if os.environ.get('ELEVENLABS_API_KEY') else '⚠️  not set (gTTS fallback)'}")
    tw = ec.config_status()
    print(f"   Twilio ready       : {'✅ yes' if tw['ready'] else '❌ no — ' + str([k for k,v in tw.items() if v == 'MISSING'])}")
    print(f"   Emergency number   : {tw['EMERGENCY_TO_NUMBER']}")
    summary = mcp.store_summary()
    print(f"   MCP store          : {summary['total_chunks']} chunks from {len(summary['documents'])} doc(s)")
    print(f"   MCP tools          : {[t['function']['name'] for t in mcp.MCP_TOOLS]}\n")
    app.run(debug=True, port=7860)
