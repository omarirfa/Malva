# Malva: Multimodal AI Interface

A fully multimodal AI interface that seamlessly integrates text, voice, and image understanding in a real-time streaming architecture.

## Overview

Malva combines multiple AI modalities into a cohesive interface:
- **Text**: Process natural language queries and generate detailed responses
- **Voice**: Convert speech to text and synthesize natural-sounding responses
- **Images**: Understand visual content and generate insights


## Multimodal Integration

Malva doesn't just offer separate channels for text, voice, and images - it integrates them:

- **Mixed-input queries**: Combine image uploads with text questions
- **Cross-modal references**: Reference visual elements in conversation
- **Unified context**: Maintain conversation history across modalities
- **Modality bridging**: Convert between speech, text, and visual understanding

## Tech Stack

- **Backend**: FastAPI with async WebSockets
- **Frontend**: Vanilla JS with optimized rendering
- **ML Models**:
  - **Text**: Gemma 3 (4B, 12B, 27B parameter variants)
  - **Speech Recognition**: Faster-Whisper with FP16 optimization
  - **Speech Synthesis**: Kokoro TTS with 50+ voice options
  - **Image Analysis**: Vision capabilities through multimodal LLM

## Implementation Details

### Multimodal Processing Pipeline

The core of Malva is its unified processing pipeline:

1. **Input Preprocessing**:
   - **Text**: Direct to LLM
   - **Voice**: Faster-Whisper ASR → Text → LLM
   - **Image**: Base64 encoding → Image embedding → LLM

2. **Inference Processing**:
   - Multimodal context assembly
   - Token streaming with chunked response
   - Parallel processing for output generation

3. **Response Generation**:
   - Real-time text streaming
   - Parallel TTS processing
   - Cached audio delivery

### Performance Optimizations

- **WebSocket Streaming**: < 50ms latency for text tokens
- **Parallel Audio Generation**: Process TTS batches concurrently
- **Caching Strategy**: TTL-based audio cache with 5000 entries
- **Dynamic Batch Sizing**: Adjust based on text complexity (5-25 sentences)
- **Worker Pools**: Dedicated thread pools for CPU-intensive tasks

## Quick Start

```bash
# Clone and install
git clone https://github.com/omarirfa/malva.git
cd malva
uv pip install -r requirements.txt
# or if on windows just run setup.bat

# Make sure you have Ollama with Gemma
ollama pull gemma3:4b

# Run it
python main.py
# Visit http://localhost:8000
```

## Configuration

Key settings in `main.py`:

```python
# Core ML models
DEFAULT_MODEL = "gemma3:4b"          # Base LLM (4B, 12B, 27B parameters)
WHISPER_MODEL_SIZE = "large-v3"      # ASR model size
WHISPER_DEVICE = "cuda"              # Hardware acceleration

# Performance settings
TTS_CACHE_SIZE = 5000                # Audio cache size
TTS_CACHE_TTL = 3600                 # Cache expiry (seconds)
TTS_BATCH_SIZE = 15                  # Audio batch size
TTS_THREAD_WORKERS = 4               # Parallel workers
```

## Multimodal Examples

Malva supports complex multimodal interactions:

- Upload an image and ask questions about specific elements
- Record voice instructions for image analysis
- Get audio responses to multimodal queries
- Maintain context across different interaction modes

## Future Enhancements

- [ ] Real-time ASR for conversational speech
- [ ] Video input processing
- [ ] Cross-modal reasoning improvements
- [ ] Prompt optimization for better multimodal understanding
- [ ] Streaming TTS generation

## License

MIT
