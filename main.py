"""FastAPI backend for an accessible multimodal AI assistant with speech and vision capabilities."""

import asyncio
import base64
import concurrent.futures
import hashlib
import io
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
import orjson
import soundfile as sf
from cachebox import TTLCache
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from kokoro import KPipeline
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration constants
# =============================================
# TTS Configuration
TTS_CACHE_SIZE = 5000
TTS_CACHE_TTL = 3600
TTS_BATCH_SIZE = 15
TTS_THREAD_WORKERS = 4
TTS_DEFAULT_LANG = "a"
TTS_DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

# API Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:4b"

# TTS Model Version
TTS_MODEL_VERSION = "1.0"

# Whisper configuration
WHISPER_MODEL_SIZE = "large-v3"  # Options: tiny, base, small, medium, large-v3
WHISPER_COMPUTE_TYPE = "float16"  # Options: float16, int8
WHISPER_DEVICE = "cuda"  # Options: cpu, cuda

# Sentence processing constants
MIN_SENTENCES_FOR_SMALL_BATCH = 5
MAX_SENTENCES_FOR_LARGE_BATCH = 50
SUCCESSFUL_BATCH_THRESHOLD = 0.5
# =============================================

# Initialize thread pools and caches
tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=TTS_THREAD_WORKERS)
asr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
pipeline_lock = threading.Lock()

# Create directories
os.makedirs("audio_cache", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create a TTS cache with TTL
TTS_CACHE = TTLCache(maxsize=TTS_CACHE_SIZE, ttl=TTS_CACHE_TTL)

# Dictionary to store TTS pipelines
tts_pipelines = {}

# Global whisper model variable
whisper_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for setup and cleanup."""
    global whisper_model

    # Startup: Initialize the default TTS pipeline
    with pipeline_lock:
        tts_pipelines[TTS_DEFAULT_LANG] = KPipeline(lang_code=TTS_DEFAULT_LANG)

    # Initialize Whisper model at startup
    logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
    )

    logger.info("Application startup complete")
    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down application...")
    tts_executor.shutdown(wait=True)
    asr_executor.shutdown(wait=True)
    logger.info("Thread pools shutdown complete")


app = FastAPI(title="Accessible Multimodal Assistant", lifespan=lifespan)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serve the main page."""
    return FileResponse("static/index.html")


@app.get("/voices")
async def get_voices():
    """Return voice data."""
    try:
        with open("static/voices.json", "rb") as f:
            voices = orjson.loads(f.read())
        return voices
    except Exception as e:
        logger.error(f"Error loading voices from API: {e}")
        return {"languages": []}


def get_tts_pipeline(lang_code):
    """Get or create a TTS pipeline for the specified language."""
    if lang_code not in tts_pipelines:
        with pipeline_lock:
            if lang_code not in tts_pipelines:  # Double-check pattern
                try:
                    tts_pipelines[lang_code] = KPipeline(lang_code=lang_code)
                except Exception as e:
                    logger.error(
                        f"Failed to create TTS pipeline for language {lang_code}: {e}"
                    )
                    # Fall back to default if available, otherwise create it
                    if TTS_DEFAULT_LANG not in tts_pipelines:
                        tts_pipelines[TTS_DEFAULT_LANG] = KPipeline(
                            lang_code=TTS_DEFAULT_LANG
                        )
                    return tts_pipelines[TTS_DEFAULT_LANG]

    return tts_pipelines[lang_code]


async def recognize_speech(audio_bytes, lang_code):
    """Convert audio bytes to text using Faster Whisper."""
    # Map our language codes to whisper language codes
    language_map = {
        "a": "en",  # American English
        "b": "en",  # British English
        "j": "ja",  # Japanese
        "z": "zh",  # Mandarin Chinese
        "e": "es",  # Spanish
        "f": "fr",  # French
        "h": "hi",  # Hindi
        "i": "it",  # Italian
        "p": "pt",  # Brazilian Portuguese
    }

    whisper_lang = language_map.get(lang_code, "en")

    try:
        # Save audio bytes to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(audio_bytes)

        try:
            # Run in thread pool to avoid blocking the main thread
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                asr_executor, _perform_whisper_recognition, temp_filename, whisper_lang
            )
            return result
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except Exception as e:
                logger.error(f"Error deleting temp file: {e}")
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        return None


def _perform_whisper_recognition(audio_file, language):
    """Actual speech recognition using Faster Whisper."""
    global whisper_model

    try:
        logger.info(f"Transcribing audio with language: {language}")

        # Simple retry mechanism if the model is not initialized
        if whisper_model is None:
            logger.info(f"Initializing Whisper model: {WHISPER_MODEL_SIZE}")
            whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )

        # Transcribe the audio
        segments, info = whisper_model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
        )

        # Collect all segments into a single text
        result = " ".join(segment.text for segment in segments)

        logger.info(f"Whisper recognized: {result}")
        logger.info(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        return result
    except Exception as e:
        logger.error(f"Whisper recognition error: {e}")
        return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming interaction."""
    await websocket.accept()

    try:
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e!s}")
                break

            message_type = data.get("type", "")

            if message_type == "text":
                # Process text input
                user_text = data.get("content", "")
                selected_model = data.get("model", DEFAULT_MODEL)

                # Get language and voice settings (with defaults)
                lang_code = data.get("lang_code", TTS_DEFAULT_LANG)
                voice = data.get("voice", TTS_DEFAULT_VOICE)
                tts_enabled = data.get("tts_enabled", True)

                # Send acknowledgment
                try:
                    await websocket.send_json(
                        {"type": "status", "content": "Processing your message..."}
                    )
                except Exception as e:
                    logger.error(f"Error sending status: {e!s}")
                    break

                # Start timing for message generation
                message_start_time = time.time()

                # Process with Gemma/Ollama
                try:
                    # Stream text response
                    text_chunks_sent = False

                    async for chunk in stream_from_gemma(
                        user_text, model=selected_model
                    ):
                        try:
                            await websocket.send_json(
                                {"type": "text_stream", "content": chunk}
                            )
                            text_chunks_sent = True
                        except Exception as e:
                            logger.error(f"Error sending text stream chunk: {e!s}")
                            break

                    # Only proceed if we successfully sent text chunks
                    if text_chunks_sent:
                        # Signal end of text streaming
                        try:
                            await websocket.send_json({"type": "text_stream_end"})
                            await websocket.send_json({"type": "status_clear"})
                        except Exception as e:
                            logger.error(f"Error signaling end of stream: {e!s}")
                            break

                        # Get complete response for TTS
                        response = await process_with_gemma(
                            user_text, model=selected_model
                        )

                        # Calculate message generation time
                        message_generation_time = time.time() - message_start_time

                        # Send metrics
                        try:
                            await websocket.send_json(
                                {
                                    "type": "metrics",
                                    "message_generation_time": round(
                                        message_generation_time, 2
                                    ),
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error sending metrics: {e!s}")
                            break

                        # If TTS is enabled, generate and send audio
                        if tts_enabled:
                            try:
                                # Indicate TTS generation has started
                                await websocket.send_json(
                                    {
                                        "type": "tts_status",
                                        "content": "Generating speech...",
                                        "progress": 0,
                                    }
                                )

                                await generate_tts(
                                    websocket, response, lang_code, voice
                                )
                            except Exception as e:
                                logger.error(f"Error during TTS: {e!s}")
                                try:
                                    await websocket.send_json(
                                        {
                                            "type": "error",
                                            "content": f"Speech generation error: {e!s}",
                                        }
                                    )
                                except Exception as e:
                                    logger.error(f"Error sending TTS error: {e!s}")
                                    break

                except Exception as e:
                    error_msg = f"Error processing text: {e!s}"
                    logger.error(error_msg)
                    try:
                        await websocket.send_json(
                            {"type": "error", "content": error_msg}
                        )
                        await websocket.send_json({"type": "status_clear"})
                    except Exception as e:
                        logger.error(f"Error sending error message: {e!s}")
                        break

            elif message_type == "audio":
                # Process audio input
                try:
                    # Get base64 audio data
                    audio_content = data.get("content", "")

                    # Handle both formats: with or without data URI prefix
                    if "," in audio_content:
                        audio_base64 = audio_content.split(",", 1)[1]
                    else:
                        audio_base64 = audio_content

                    # Get language and model settings
                    selected_model = data.get("model", DEFAULT_MODEL)
                    lang_code = data.get("lang_code", TTS_DEFAULT_LANG)
                    voice = data.get("voice", TTS_DEFAULT_VOICE)
                    tts_enabled = data.get("tts_enabled", True)

                    # Send status
                    await websocket.send_json(
                        {"type": "status", "content": "Processing your voice..."}
                    )

                    # Convert to audio bytes
                    audio_bytes = base64.b64decode(audio_base64)

                    # Use Whisper to convert speech to text
                    recognized_text = await recognize_speech(audio_bytes, lang_code)

                    if recognized_text:
                        logger.info(f"Speech recognized: {recognized_text}")

                        # Send recognized text back to client
                        await websocket.send_json(
                            {"type": "recognition", "content": recognized_text}
                        )

                        # Clear processing status but DON'T process yet - wait for confirmation
                        await websocket.send_json({"type": "status_clear"})

                        # Don't automatically process with the LLM - client will send a text message
                        # when user confirms

                    else:
                        logger.warning("Could not recognize speech")
                        await websocket.send_json(
                            {
                                "type": "error",
                                "content": "Could not recognize speech. Please try again.",
                            }
                        )
                        await websocket.send_json({"type": "status_clear"})

                except Exception as e:
                    error_msg = f"Error processing audio: {e!s}"
                    logger.error(error_msg)
                    await websocket.send_json({"type": "error", "content": error_msg})
                    await websocket.send_json({"type": "status_clear"})

            elif message_type == "multimodal":
                try:
                    # Process text + image
                    user_text = data.get("text", "")

                    # Extract image data efficiently
                    image_b64 = None
                    image_data = None

                    # Check if we already have a base64 string
                    image_src = data.get("image", "")
                    if (
                        image_src
                        and isinstance(image_src, str)
                        and image_src.startswith("data:image")
                    ):
                        try:
                            # Extract the base64 part after the comma
                            image_parts = image_src.split(",", 1)
                            if len(image_parts) > 1:
                                image_b64 = image_parts[1]
                                image_data = base64.b64decode(image_b64)
                        except Exception as e:
                            logger.error(f"Error decoding image: {e!s}")
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "content": f"Error processing image: {e!s}",
                                }
                            )
                            continue

                    # Get language and voice settings (with defaults)
                    lang_code = data.get("lang_code", TTS_DEFAULT_LANG)
                    voice = data.get("voice", TTS_DEFAULT_VOICE)
                    tts_enabled = data.get("tts_enabled", True)

                    # Send acknowledgment
                    await websocket.send_json(
                        {
                            "type": "status",
                            "content": "Processing your image and message...",
                        }
                    )

                    # Send image info back for chat display
                    if image_src:
                        await websocket.send_json(
                            {"type": "image_info", "image_src": image_src}
                        )

                    # Start timing for message generation
                    message_start_time = time.time()

                    # Get AI response with streaming
                    selected_model = data.get("model", DEFAULT_MODEL)

                    # Process with Gemma/Ollama
                    text_chunks_sent = False

                    async for chunk in stream_from_gemma(
                        user_text, image_data, image_b64, model=selected_model
                    ):
                        try:
                            await websocket.send_json(
                                {"type": "text_stream", "content": chunk}
                            )
                            text_chunks_sent = True
                        except Exception as e:
                            logger.error(f"Error sending text stream chunk: {e!s}")
                            break

                    # Only proceed if we successfully sent text chunks
                    if text_chunks_sent:
                        # Signal end of text streaming
                        try:
                            await websocket.send_json({"type": "text_stream_end"})
                            await websocket.send_json({"type": "status_clear"})
                        except Exception as e:
                            logger.error(f"Error signaling end of stream: {e!s}")
                            break

                        # Get complete response for TTS
                        response = await process_with_gemma(
                            user_text, image_data, image_b64, model=selected_model
                        )

                        # Calculate message generation time
                        message_generation_time = time.time() - message_start_time

                        # Send metrics
                        try:
                            await websocket.send_json(
                                {
                                    "type": "metrics",
                                    "message_generation_time": round(
                                        message_generation_time, 2
                                    ),
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error sending metrics: {e!s}")
                            break

                        # If TTS is enabled, generate and send audio
                        if tts_enabled:
                            try:
                                # Indicate TTS generation has started
                                await websocket.send_json(
                                    {
                                        "type": "tts_status",
                                        "content": "Generating speech...",
                                        "progress": 0,
                                    }
                                )

                                await generate_tts(
                                    websocket, response, lang_code, voice
                                )
                            except Exception as e:
                                logger.error(f"Error during TTS: {e!s}")
                                try:
                                    await websocket.send_json(
                                        {
                                            "type": "error",
                                            "content": f"Speech generation error: {e!s}",
                                        }
                                    )
                                except Exception as e:
                                    logger.error(f"Error sending TTS error: {e!s}")
                                    break

                except Exception as e:
                    error_msg = f"Error processing multimodal request: {e!s}"
                    logger.error(error_msg)
                    try:
                        await websocket.send_json(
                            {"type": "error", "content": error_msg}
                        )
                        await websocket.send_json({"type": "status_clear"})
                    except Exception as e:
                        logger.error(f"Error sending error message: {e!s}")
                        break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        error_message = f"WebSocket error: {e!s}"
        logger.error(error_message)
        try:
            await websocket.send_json({"type": "error", "content": error_message})
            await websocket.send_json({"type": "status_clear"})
        except Exception as e:
            logger.error(f"Error sending WebSocket error: {e!s}")


async def stream_from_gemma(
    text: str, image_data=None, image_b64=None, model=DEFAULT_MODEL
):
    """Stream responses from Gemma 3 model via Ollama."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"model": model, "prompt": text, "stream": True}

            # Add image if provided, using pre-encoded base64 when available
            if image_data:
                try:
                    # Use pre-encoded base64 if available
                    if image_b64:
                        payload["images"] = [image_b64]
                    else:
                        # Process the image only if needed
                        img = Image.open(io.BytesIO(image_data))
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        payload["images"] = [img_b64]
                except Exception as e:
                    logger.error(f"Image processing error: {e}")
                    raise

            try:
                async with client.stream(
                    "POST", OLLAMA_API_URL, json=payload, timeout=None
                ) as response:
                    response.raise_for_status()

                    async for chunk in response.aiter_lines():
                        if not chunk:
                            continue

                        try:
                            chunk_data = orjson.loads(chunk)
                            if "response" in chunk_data:
                                yield chunk_data["response"]
                        except Exception as e:
                            logger.error(f"Chunk processing error: {e}")
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error during Ollama streaming: {e.response.status_code} - {e.response.text}"
                )
                yield f"Sorry, the model service returned an error: {e.response.status_code}"
            except httpx.RequestError as e:
                logger.error(f"Request error during Ollama streaming: {e}")
                yield "Sorry, I encountered a connection error with the model service."

    except Exception as e:
        logger.error(f"Ollama streaming API error: {e}")
        yield f"Sorry, I encountered an error: {e!s}"


async def process_with_gemma(
    text: str, image_data=None, image_b64=None, model=DEFAULT_MODEL
) -> str:
    """Process input with Gemma 3 model via Ollama (non-streaming)."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"model": model, "prompt": text, "stream": False}

            # Add image if provided, using pre-encoded base64 when available
            if image_data:
                try:
                    # Use pre-encoded base64 if available
                    if image_b64:
                        payload["images"] = [image_b64]
                    else:
                        # Process the image only if needed
                        img = Image.open(io.BytesIO(image_data))
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        payload["images"] = [img_b64]
                except Exception as e:
                    logger.error(f"Image processing error: {e}")
                    raise

            try:
                response = await client.post(OLLAMA_API_URL, json=payload)
                response.raise_for_status()
                return orjson.loads(response.content).get(
                    "response", "Sorry, I couldn't process that request."
                )
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error during Ollama request: {e.response.status_code} - {e.response.text}"
                )
                return f"Sorry, the model service returned an error: {e.response.status_code}"
            except httpx.RequestError as e:
                logger.error(f"Request error during Ollama request: {e}")
                return "Sorry, I encountered a connection error with the model service."

    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return f"Sorry, I encountered an error: {e!s}"


# Function to strip markdown formatting for TTS
def strip_markdown(text):
    """Strip markdown formatting for TTS to get plain text."""
    # Simple regex-based strip
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Italic
    text = re.sub(r"__(.*?)__", r"\1", text)  # Bold
    text = re.sub(r"_(.*?)_", r"\1", text)  # Italic
    text = re.sub(r"#{1,6}\s+(.*?)$", r"\1", text, flags=re.MULTILINE)  # Headers
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL)  # Code blocks
    text = re.sub(r"`(.*?)`", r"\1", text)  # Inline code
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)  # Links
    text = re.sub(r"^\s*[-*+]\s+(.*?)$", r"\1", text, flags=re.MULTILINE)  # Lists
    text = re.sub(
        r"^\s*\d+\.\s+(.*?)$", r"\1", text, flags=re.MULTILINE
    )  # Numbered lists
    return text


def get_tts_cache_key(text, voice, lang_code):
    """Generate a cache key for TTS audio."""
    key_str = f"{text}|{voice}|{lang_code}|{TTS_MODEL_VERSION}"
    return hashlib.md5(key_str.encode()).hexdigest()


# Function to process a batch of sentences with TTS
def process_tts_batch(batch_text, voice, pipeline):
    """Process a batch of text with TTS in a separate thread."""
    try:
        audio_chunks = []
        generator = pipeline(batch_text, voice=voice, speed=1.0)
        for _, _, audio in generator:
            audio_chunks.append(audio)
        return np.concatenate(audio_chunks) if audio_chunks else np.array([])
    except Exception as e:
        logger.error(f"Error processing TTS batch: {e}")
        # Return empty array on error
        return np.array([])


async def generate_tts(
    websocket, text, lang_code=TTS_DEFAULT_LANG, voice=TTS_DEFAULT_VOICE
):
    """Generate a single TTS audio for the entire response with markdown stripped."""
    try:
        # Start timing for TTS generation
        tts_start_time = time.time()

        # Strip markdown formatting for TTS
        clean_text = strip_markdown(text)

        # Create cache key for entire response
        full_text_cache_key = get_tts_cache_key(clean_text, voice, lang_code)

        # Check if full response is in cache
        if full_text_cache_key in TTS_CACHE:
            try:
                audio_b64, audio_duration = TTS_CACHE[full_text_cache_key]

                # Update status
                await websocket.send_json(
                    {
                        "type": "tts_status",
                        "content": "Using cached audio...",
                        "progress": 100,
                    }
                )

                # Clear TTS status
                await websocket.send_json({"type": "tts_status_clear"})

                # Send cached audio
                await websocket.send_json(
                    {
                        "type": "audio_full",
                        "data": audio_b64,
                        "duration": audio_duration,
                        "tts_generation_time": 0.01,  # Very fast for cache hit
                        "cached": True,  # Optional flag for metrics
                    }
                )
                return
            except Exception as e:
                # In case of any cache error, just regenerate
                logger.error(f"Cache retrieval error: {e}")
                # Continue with generation

        # Get pipeline for the language
        pipeline = get_tts_pipeline(lang_code)
        loop = asyncio.get_running_loop()

        # Split text into reasonable paragraphs first, then sentences
        paragraphs = re.split(r"\n\s*\n", clean_text)
        all_sentences = []

        for paragraph in paragraphs:
            # Split sentences but preserve paragraph boundaries
            sentences = re.split(r"([.!?।]\s+)", paragraph)

            # Recombine sentences with their punctuation
            i = 0
            while i < len(sentences) - 1:
                if i + 1 < len(sentences) and re.match(r"[.!?।]\s+", sentences[i + 1]):
                    sentences[i] = sentences[i] + sentences[i + 1]
                    sentences.pop(i + 1)
                else:
                    i += 1

            # Add non-empty sentences to our list
            for sentence in sentences:
                if sentence.strip():
                    all_sentences.append(sentence.strip())

        # Add a short pause between paragraphs
        all_sentences = [s for s in all_sentences if s.strip()]
        total_sentences = len(all_sentences)

        # Process in larger batches for better performance
        batch_size = TTS_BATCH_SIZE

        # If the text is very short, use a smaller batch size
        if total_sentences < MIN_SENTENCES_FOR_SMALL_BATCH:
            batch_size = max(1, total_sentences)
        # If the text is very long, use a larger batch size to reduce overhead
        elif total_sentences > MAX_SENTENCES_FOR_LARGE_BATCH:
            batch_size = min(25, total_sentences // 2)

        # Calculate total batches for progress reporting
        total_batches = (len(all_sentences) + batch_size - 1) // batch_size

        # Process batches in parallel using the thread pool
        # Each batch gets its own task in the thread pool
        futures = []
        batch_texts = []
        successful_batches = 0
        failed_batches = 0
        all_audio = []

        # Prepare batches
        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i : i + batch_size]
            if not batch:
                continue

            # Join the batch with appropriate separators
            batch_text = " ".join(batch)
            batch_texts.append(batch_text)

            # Check cache for this batch
            batch_cache_key = get_tts_cache_key(batch_text, voice, lang_code)

            if batch_cache_key in TTS_CACHE:
                try:
                    # Use cached audio for this batch
                    audio_b64, _ = TTS_CACHE[batch_cache_key]
                    # Convert back from base64 to numpy array
                    audio_bytes = base64.b64decode(audio_b64)
                    with io.BytesIO(audio_bytes) as buf:
                        audio_array, _ = sf.read(buf)
                    all_audio.append(audio_array)
                    successful_batches += 1

                    # Report progress
                    progress = min(
                        100, int(((i // batch_size) + 1) / total_batches * 100)
                    )
                    await websocket.send_json(
                        {
                            "type": "tts_status",
                            "content": f"Generating speech... {progress}% (using cache)",
                            "progress": progress,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error retrieving cached audio batch: {e}")
                    # If we fail to retrieve from cache, generate it
                    future = loop.run_in_executor(
                        tts_executor, process_tts_batch, batch_text, voice, pipeline
                    )
                    futures.append(
                        (future, batch_cache_key, batch_text, i // batch_size)
                    )
            else:
                # Need to generate this batch
                future = loop.run_in_executor(
                    tts_executor, process_tts_batch, batch_text, voice, pipeline
                )
                futures.append((future, batch_cache_key, batch_text, i // batch_size))

        # Process futures as they complete
        for _, (future, batch_cache_key, _, batch_idx) in enumerate(futures):
            try:
                # Get audio for this batch
                audio = await future

                if len(audio) > 0:
                    all_audio.append(audio)
                    successful_batches += 1

                    # Cache this batch's audio
                    try:
                        with io.BytesIO() as buf:
                            sf.write(buf, audio, SAMPLE_RATE, format="WAV")
                            buf.seek(0)
                            audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                        # Calculate duration
                        batch_duration = len(audio) / SAMPLE_RATE

                        # Store in cache
                        TTS_CACHE[batch_cache_key] = (audio_b64, batch_duration)
                    except Exception as e:
                        logger.error(f"Error caching audio batch: {e}")
                        # Continue even if caching fails
                else:
                    failed_batches += 1

                # Report progress
                progress = min(100, int((batch_idx + 1) / total_batches * 100))
                await websocket.send_json(
                    {
                        "type": "tts_status",
                        "content": f"Generating speech... {progress}%",
                        "progress": progress,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing TTS batch: {e}")
                failed_batches += 1
                # Continue with other batches

        # Check if we have a partial success (some batches failed but some succeeded)
        total_processed = successful_batches + failed_batches
        if total_processed > 0 and failed_batches > 0:
            success_rate = successful_batches / total_processed
            # If more than half the batches failed, warn the user
            if success_rate < SUCCESSFUL_BATCH_THRESHOLD:
                await websocket.send_json(
                    {
                        "type": "status",
                        "content": f"Warning: Only {int(success_rate * 100)}% of speech successfully generated",
                    }
                )

        # Final progress update
        await websocket.send_json(
            {"type": "tts_status", "content": "Processing audio...", "progress": 100}
        )

        # Concatenate all audio chunks
        if all_audio:
            try:
                # Combine audio arrays efficiently
                full_audio = np.concatenate(all_audio)

                # Get audio duration in seconds
                audio_duration = len(full_audio) / SAMPLE_RATE

                # Convert numpy array to WAV bytes
                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, full_audio, SAMPLE_RATE, format="WAV")
                audio_bytes.seek(0)

                # Calculate TTS generation time
                tts_generation_time = time.time() - tts_start_time

                # Clear TTS status
                await websocket.send_json({"type": "tts_status_clear"})

                # Encode full audio
                audio_b64 = base64.b64encode(audio_bytes.getvalue()).decode("utf-8")

                # Try to cache the complete response
                try:
                    TTS_CACHE[full_text_cache_key] = (
                        audio_b64,
                        round(audio_duration, 2),
                    )
                except Exception as e:
                    logger.error(f"Error caching full audio: {e}")
                    # Continue even if full caching fails

                # Send base64 encoded audio and duration info
                await websocket.send_json(
                    {
                        "type": "audio_full",
                        "data": audio_b64,
                        "duration": round(audio_duration, 2),
                        "tts_generation_time": round(tts_generation_time, 2),
                    }
                )
            except ValueError as e:
                # This likely means audio arrays have inconsistent dimensions
                error_msg = f"Error combining audio: {e!s}"
                logger.error(error_msg)
                await websocket.send_json({"type": "error", "content": error_msg})
                await websocket.send_json({"type": "tts_status_clear"})

        else:
            await websocket.send_json(
                {"type": "error", "content": "TTS error: No audio generated"}
            )
            await websocket.send_json({"type": "tts_status_clear"})

    except Exception as e:
        logger.error(f"TTS error: {e}")
        await websocket.send_json({"type": "error", "content": f"TTS error: {e!s}"})
        await websocket.send_json({"type": "tts_status_clear"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
