import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from lab_processing import process_l_channel, clahe_enhance, bilateral_enhance, face_enhance
from metrics import compute_metrics

app = FastAPI(title="Face Enhancement API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

ENHANCE_MODES = {
    "clahe":     lambda img: process_l_channel(img, enhance_fn=lambda l: clahe_enhance(l, clip_limit=0.02)),
    "bilateral": lambda img: process_l_channel(img, enhance_fn=lambda l: bilateral_enhance(l, smoothing_degree=0.01)),
    "combined":  face_enhance,
}


MAX_INPUT_DIM = 1200  # Boyutu sinirla: Render free tier RAM + iOS Safari data URL limiti


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Gecersiz goruntu dosyasi.")

    h, w = img.shape[:2]
    if max(h, w) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def encode_image(img: np.ndarray, ext: str = ".jpg") -> bytes:
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise HTTPException(status_code=500, detail="Goruntu kodlanamadi.")
    return buf.tobytes()


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    mode: str = Query(default="clahe", enum=list(ENHANCE_MODES.keys())),
):
    """
    Goruntu yukle ve iyilestirilmis versiyonu indir.

    - **file**: JPG / PNG goruntu
    - **mode**: `clahe` | `bilateral` | `combined`
    """
    raw = await file.read()
    original = decode_image(raw)

    enhanced = ENHANCE_MODES[mode](original)
    output   = encode_image(enhanced, ext=".jpg")

    return StreamingResponse(
        io.BytesIO(output),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=enhanced_{file.filename}"},
    )


@app.post("/enhance/metrics")
async def enhance_with_metrics(
    file: UploadFile = File(...),
    mode: str = Query(default="clahe", enum=list(ENHANCE_MODES.keys())),
):
    """
    Goruntu yukle; iyilestirilmis goruntu (base64) + SSIM/PSNR metriklerini JSON olarak al.

    - **file**: JPG / PNG goruntu
    - **mode**: `clahe` | `bilateral` | `combined`
    """
    import base64

    raw = await file.read()
    original = decode_image(raw)

    enhanced = ENHANCE_MODES[mode](original)

    # Pipeline goruntuyu buyutmus olabilir (super-resolution). SSIM/PSNR ayni
    # boyut ister, bu yuzden orijinali bicubic ile cikti boyutuna esitliyoruz.
    if original.shape != enhanced.shape:
        original_for_metrics = cv2.resize(
            original,
            (enhanced.shape[1], enhanced.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        original_for_metrics = original

    metrics = compute_metrics(original_for_metrics, enhanced)
    output  = encode_image(enhanced, ext=".jpg")

    return {
        "mode":            mode,
        "original_size":   {"width": original.shape[1], "height": original.shape[0]},
        "metrics":         metrics,
        "enhanced_image":  base64.b64encode(output).decode("utf-8"),
    }
