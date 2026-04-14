"""
server.py — Flask backend for MediBot (MCP edition).

Endpoints:
  GET  /                  → serve index.html
  GET  /api/health        → key + store status
  GET  /api/docs          → list loaded documents & chunk count
  POST /api/upload-doc    → upload a new PDF or TXT into the MCP store
  POST /api/analyze       → multipart: audio (optional) + image (optional) → JSON response
  GET  /api/audio/<name>  → serve generated audio files

Architecture change from RAG edition:
  • rag_engine.py  →  mcp_server.py
  • Context is NOT injected upfront. Instead, the LLM receives MCP_TOOLS and
    decides autonomously whether/when to call search_medical_knowledge,
    list_documents, or get_document_summary.
  • The agentic_complete() function runs the full tool-calling loop until the
    model stops issuing tool calls and returns a final text response.
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

    typed_text = request.form.get("text", "").strip()
    user_query = (speech_text + " " + typed_text).strip() or "Please analyze my image."

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

    # 4. TTS
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
        "speech_text":     speech_text,
        "doctor_response": doctor_response,
        "audio_url":       audio_url,
        "mcp_tools_used":  [t["tool"] for t in tool_log],
        "tool_call_count": len(tool_log),
    })


@app.route("/api/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(str(AUDIO_DIR), filename)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    static_dir = Path("static")
    target = static_dir / path if path else static_dir / "index.html"
    if target.exists():
        return send_file(str(target))
    return send_file(str(static_dir / "index.html"))


if __name__ == "__main__":
    print("\n🩺  MediBot MCP server starting...")
    print(f"   GROQ_API_KEY      : {'✅ set' if GROQ_API_KEY else '❌ missing'}")
    print(f"   ELEVENLABS_API_KEY: {'✅ set' if os.environ.get('ELEVENLABS_API_KEY') else '⚠️  not set (gTTS fallback)'}")
    summary = mcp.store_summary()
    print(f"   MCP store         : {summary['total_chunks']} chunks from {len(summary['documents'])} doc(s)")
    print(f"   MCP tools         : {[t['function']['name'] for t in mcp.MCP_TOOLS]}\n")
    app.run(debug=True, port=7860)
