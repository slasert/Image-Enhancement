import os
import urllib.request

import cv2
import numpy as np


def process_l_channel(image: np.ndarray, enhance_fn=None) -> np.ndarray:
    """
    Görüntüyü BGR -> LAB formatına çevirir,
    yalnızca L kanalına işlem uygular ve BGR'ye geri döner.

    Args:
        image:      BGR formatında uint8 görüntü (H x W x 3)
        enhance_fn: L kanalına uygulanacak fonksiyon f(l: np.ndarray) -> np.ndarray
                    None ise L kanalı değiştirilmeden döner.

    Returns:
        İşlenmiş BGR görüntüsü (uint8)
    """
    if image is None or image.ndim != 3:
        raise ValueError("Geçerli bir BGR görüntüsü (H x W x 3) giriniz.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if enhance_fn is not None:
        l = enhance_fn(l)
        l = np.clip(l, 0, 255).astype(np.uint8)

    lab_processed = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)


def clahe_enhance(l: np.ndarray, clip_limit: float = 0.02, tile_size: int = 8) -> np.ndarray:
    scaled_clip = clip_limit * 255
    clahe = cv2.createCLAHE(clipLimit=scaled_clip, tileGridSize=(tile_size, tile_size))
    return clahe.apply(l)


def bilateral_enhance(l: np.ndarray, smoothing_degree: float = 0.01) -> np.ndarray:
    sigma = smoothing_degree * 255
    d = max(3, int(sigma * 3) | 1)
    return cv2.bilateralFilter(l, d=d, sigmaColor=sigma, sigmaSpace=sigma)


# ---------------------------------------------------------------------------
# Face-focused enhancement
# Goal: turn a blurry / low-light face crop into a sharper, larger image
# that is easier to recognize. Uses OpenCV's dnn_superres (FSRCNN x2) for
# AI-based upscaling - small (~38 KB), fast, suitable for low-RAM hosts
# like Render's free tier. Falls back to INTER_CUBIC if the model is missing.
# ---------------------------------------------------------------------------

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_SR_MODEL_URLS = {
    2: "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    3: "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
    4: "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
}
_sr_model_cache: dict = {}


def _ensure_sr_model(scale: int) -> str:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, f"FSRCNN_x{scale}.pb")
    if not os.path.exists(path):
        url = _SR_MODEL_URLS[scale]
        print(f"[face_enhance] FSRCNN_x{scale} modeli indiriliyor (~40 KB)...")
        urllib.request.urlretrieve(url, path)
        print(f"[face_enhance] Model hazir: {path}")
    return path


def _sr_upscale(image: np.ndarray, scale: int = 2) -> np.ndarray:
    if scale not in _sr_model_cache:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_ensure_sr_model(scale))
        sr.setModel("fsrcnn", scale)
        _sr_model_cache[scale] = sr
    return _sr_model_cache[scale].upsample(image)


def face_enhance(image: np.ndarray, sr_scale: int = 2) -> np.ndarray:
    """
    Yuz tanimayi kolaylastirmak icin tasarlanmis pipeline:
    1. Renk-duyarli hafif denoising (NLM)
    2. LAB - L kanalinda CLAHE (renkleri korur, kontrasti artirir)
    3. Hafif unsharp mask (yumusak keskinlestirme)
    4. AI super-resolution (FSRCNN x2) - piksel sayisini 4 katina cikarir
    Model yoksa veya yuklenemezse INTER_CUBIC ile geri dusulur.
    """
    if image is None or image.ndim != 3:
        raise ValueError("Gecerli bir BGR goruntusu (H x W x 3) giriniz.")

    denoised = cv2.fastNlMeansDenoisingColored(
        image, None,
        h=5, hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.4, blur, -0.4, 0)

    try:
        return _sr_upscale(sharpened, scale=sr_scale)
    except Exception as e:
        print(f"[face_enhance] DNN SR basarisiz, INTER_CUBIC kullaniliyor: {e}")
        h, w = sharpened.shape[:2]
        return cv2.resize(sharpened, (w * sr_scale, h * sr_scale),
                          interpolation=cv2.INTER_CUBIC)
