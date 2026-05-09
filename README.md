---
title: Image Enhancement
emoji: 📷
sdk: docker
app_port: 7860
colorFrom: indigo
colorTo: purple
pinned: false
---

# 🖼️ Image Enhancement API

> A FastAPI-powered image enhancement application that processes images in **LAB color space** using classical computer vision techniques — no deep learning required.

> ⚠️ **Warning:** This app is designed for **desktop / computer use only**. The UI is not optimized for mobile devices — please open it on a computer for the best experience.

**Live demo:** https://sila0-image-enhancement.hf.space

---

## Overview

This application exposes a REST API and a browser-based UI for enhancing low-quality images. It applies advanced image processing algorithms in the **LAB color space**, operating only on the **L (luminance) channel** to preserve natural colors while improving brightness, contrast, and sharpness.

Three enhancement modes are available, ranging from quick contrast correction to a full 6-stage professional pipeline.

---

## Enhancement Modes

### `clahe` — Contrast Limited Adaptive Histogram Equalization
Divides the image into small tiles and equalizes the histogram of each tile independently. The `clipLimit` parameter prevents over-amplification of noise. Ideal for images with uneven lighting.

### `bilateral` — Bilateral Filtering
A non-linear smoothing filter that preserves edges while reducing noise. Unlike Gaussian blur, it considers both spatial proximity and pixel intensity similarity — producing a "clean" look without losing structure.

### `combined` — Face-Focused Enhancement Pipeline
Designed for blurry / low-light face crops. Four stages:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | **Non-local Means Denoising** (color-aware) | Remove noise while preserving texture |
| 2 | **CLAHE on L channel** | Boost contrast without shifting hue |
| 3 | **Unsharp mask** | Pull out edge / contour detail |
| 4 | **FSRCNN x2 super-resolution** | 2× upscale via a tiny (~38 KB) ONNX-style model; falls back to bicubic if the model file is missing |

The FSRCNN weights are downloaded on first use to `models/FSRCNN_x2.pb`.

---

## API Endpoints

### `POST /enhance`
Upload an image and download the enhanced version.

**Parameters:**
- `file` — JPG or PNG image
- `mode` — `clahe` | `bilateral` | `combined` (default: `clahe`)

**Response:** Enhanced image as JPEG download.

---

### `POST /enhance/metrics`
Upload an image and receive the enhanced image + quality metrics in JSON.

**Response:**
```json
{
  "mode": "clahe",
  "original_size": { "width": 1920, "height": 1080 },
  "metrics": {
    "ssim": 0.874,
    "psnr": 32.5
  },
  "enhanced_image": "<base64 encoded JPEG>"
}
```

**Metrics explained:**
- **SSIM** (Structural Similarity Index) — measures perceptual similarity between original and enhanced image. Range: 0–1, higher = more similar structure preserved.
- **PSNR** (Peak Signal-to-Noise Ratio) — measures reconstruction quality in dB. Higher values indicate less distortion introduced by enhancement.

---

### `POST /chat`
Chat with the **Image Enhancement Assistant** — an AI guide that explains techniques (CLAHE, MSRCR, FFT, LAB color space), interprets metrics, and recommends the best mode for a given image.

**Request body (JSON):**
```json
{
  "messages": [
    { "role": "user", "content": "Why is my SSIM so low?" }
  ],
  "context": "Mode used: combined | PSNR: 28.4 dB | SSIM: 0.71"
}
```

- `messages` — chat history, alternating `user` / `assistant`
- `context` — optional session context (current mode, metrics) so the assistant can answer with concrete numbers

**Response:** plain-text **streamed** response (`text/plain; charset=utf-8`). The assistant runs on Anthropic's Claude API. Set `ANTHROPIC_API_KEY` either in a `.env` file (recommended — see `.env.example`) or as a shell environment variable.

The browser UI exposes this as a floating speech-bubble button in the bottom-right corner — click it to open the chat panel. After enhancing an image, the panel automatically receives session context (current mode + metrics) so the assistant can answer with concrete numbers.

---

## LAB Color Space

All processing is performed in **CIE LAB** color space:

```
BGR  →  LAB  →  [ enhance L channel only ]  →  BGR
```

- **L** — Lightness (0 = black, 255 = white) — *this is what we modify*
- **A** — Green ↔ Red color axis
- **B** — Blue ↔ Yellow color axis

By isolating the L channel, enhancements affect only brightness and contrast — color hues remain untouched, preventing the color distortion common in naive histogram equalization.

---

## Project Structure

```
my ai proje/
├── main.py            # FastAPI app — routes and endpoint logic
├── lab_processing.py  # All image processing algorithms
├── metrics.py         # SSIM / PSNR computation
├── chat.py            # Claude-powered Image Enhancement Assistant
├── requirements.txt   # Python dependencies
├── start.bat          # One-click launcher (Windows)
└── static/
    ├── index.html     # Browser UI (incl. chat panel)
    ├── script.js      # Frontend logic
    └── style.css      # Styling
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| FastAPI | REST API framework |
| OpenCV | Image processing (CLAHE, bilateral, FFT, etc.) |
| NumPy | Array operations and mathematical transforms |
| scikit-image | SSIM metric computation |
| Anthropic SDK | Claude API for the in-app Image Enhancement Assistant |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/slasert/image-enhancement.git
cd image-enhancement

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key (required for /chat — the AI assistant)
#    Easiest: copy .env.example to .env and put your key in it.
cp .env.example .env
#    Then edit .env and replace the placeholder. python-dotenv loads it
#    automatically at server startup.
#
#    Or export it as a shell env var instead:
#      macOS / Linux:   export ANTHROPIC_API_KEY=sk-ant-...
#      Windows (cmd):   set ANTHROPIC_API_KEY=sk-ant-...
#      Windows (PS):    $env:ANTHROPIC_API_KEY = "sk-ant-..."

# 4. Start the server
uvicorn main:app --reload

# 5. Open in browser
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive API docs)
```

Or on Windows, simply double-click **`start.bat`**. (The `/chat` endpoint will return an error message until `ANTHROPIC_API_KEY` is set; the rest of the app works without it.)

---

## Interactive API Docs

FastAPI automatically generates interactive documentation at:
- **Swagger UI** → `http://localhost:8000/docs`
- **ReDoc** → `http://localhost:8000/redoc`

You can test all endpoints directly from the browser without any additional tools.

---

*Built with OpenCV and FastAPI — classical computer vision, no deep learning required.*
