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

    # BGR -> LAB (L: 0-255, A: 0-255, B: 0-255 — uint8 aralığı)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Kanalları ayır
    l, a, b = cv2.split(lab)

    # Yalnızca L kanalına işlem uygula
    if enhance_fn is not None:
        l = enhance_fn(l)
        l = np.clip(l, 0, 255).astype(np.uint8)

    # Kanalları birleştir ve BGR'ye geri çevir
    lab_processed = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)

    return result


# --- Hazır iyileştirme fonksiyonları ---

def clahe_enhance(l: np.ndarray, clip_limit: float = 0.02, tile_size: int = 8) -> np.ndarray:
    scaled_clip = clip_limit * 255
    clahe = cv2.createCLAHE(clipLimit=scaled_clip, tileGridSize=(tile_size, tile_size))
    return clahe.apply(l)


def bilateral_enhance(l: np.ndarray, smoothing_degree: float = 0.01) -> np.ndarray:
    sigma = smoothing_degree * 255
    d = max(3, int(sigma * 3) | 1)
    return cv2.bilateralFilter(l, d=d, sigmaColor=sigma, sigmaSpace=sigma)


def gamma_enhance(l: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(l, table)


def histogram_equalize(l: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(l)


def smart_enhance(l: np.ndarray) -> np.ndarray:
    mean_val = float(np.mean(l))
    if mean_val > 0:
        scale = min(120.0 / mean_val, 2.5)
        l = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    blur = cv2.GaussianBlur(l, (0, 0), 1.0)
    l = cv2.addWeighted(l, 1.7, blur, -0.7, 0)
    return np.clip(l, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Multi-Scale Retinex with Color Restoration (MSRCR)
# Used in NASA image processing & professional photo software.
# Simultaneously fixes: dark/uneven lighting, color casts, low contrast.
# ---------------------------------------------------------------------------

def _ssr(img_f32: np.ndarray, sigma: float) -> np.ndarray:
    """Single Scale Retinex in log domain."""
    blurred = cv2.GaussianBlur(img_f32, (0, 0), sigma)
    return np.log(img_f32 + 1.0) - np.log(np.maximum(blurred, 1e-6) + 1.0)


def _msrcr(image: np.ndarray,
           sigmas: tuple = (15, 80, 250),
           color_gain: float = 128.0,
           color_offset: float = 128.0,
           restore_factor: float = 125.0) -> np.ndarray:
    """Multi-Scale Retinex with Color Restoration."""
    img_f = image.astype(np.float32) + 1.0

    # Multi-scale retinex: average SSR across scales
    msr = np.zeros_like(img_f)
    for s in sigmas:
        msr += _ssr(img_f, s)
    msr /= len(sigmas)

    # Color restoration function
    img_sum = np.sum(img_f, axis=2, keepdims=True)
    color_restore = restore_factor * (
        np.log(img_f) - np.log(np.maximum(img_sum, 1e-6))
    )

    msrcr_out = msr * color_restore

    # Per-channel normalization → output centered at 128
    result = np.zeros_like(img_f)
    for c in range(3):
        ch = msrcr_out[:, :, c]
        mean, std = float(ch.mean()), float(ch.std())
        result[:, :, c] = color_gain * (ch - mean) / (std + 1e-6) + color_offset

    return np.clip(result, 0, 255).astype(np.uint8)


def _percentile_stretch(image: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Stretch each channel so p_lo → 0 and p_hi → 255 for max dynamic range."""
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        ch = image[:, :, c].astype(np.float32)
        p_lo, p_hi = np.percentile(ch, lo), np.percentile(ch, hi)
        if p_hi > p_lo:
            out[:, :, c] = (ch - p_lo) / (p_hi - p_lo) * 255.0
        else:
            out[:, :, c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def _freq_sharpen(image: np.ndarray, strength: float = 1.8) -> np.ndarray:
    """High-boost filter in FFT domain — amplifies ALL detail frequencies."""
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    d = np.sqrt(x * x + y * y)
    sigma = min(rows, cols) * 0.07
    low_pass = np.exp(-d ** 2 / (2 * sigma ** 2))
    hb_filter = 1.0 + strength * (1.0 - low_pass)   # high-boost = 1 + k*(1-LP)

    out = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        ch = image[:, :, c].astype(np.float32)
        f = np.fft.fftshift(np.fft.fft2(ch))
        ch_back = np.real(np.fft.ifft2(np.fft.ifftshift(f * hb_filter)))
        out[:, :, c] = ch_back
    return np.clip(out, 0, 255).astype(np.uint8)


def _adaptive_sharpen(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Sharpen proportionally to edge strength — flat areas stay smooth."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    edge = cv2.GaussianBlur(edge, (0, 0), 2.0)
    edge = (edge / (edge.max() + 1e-6)).astype(np.float32)
    edge3 = np.stack([edge] * 3, axis=2)

    img_f = image.astype(np.float32)
    blur  = cv2.GaussianBlur(image, (0, 0), 1.0).astype(np.float32)
    sharp = img_f + strength * (img_f - blur)

    result = sharp * edge3 + img_f * (1.0 - edge3)
    return np.clip(result, 0, 255).astype(np.uint8)


def full_enhance(image: np.ndarray) -> np.ndarray:
    """
    Maximum quality — no deep learning:
      1. Color-aware denoising   (fastNlMeansDenoisingColored)
      2. MSRCR                   (illumination + color restoration)
      3. Percentile stretch      (maximize full dynamic range)
      4. Frequency-domain boost  (FFT high-boost — all detail scales)
      5. Adaptive edge sharpen   (sharpen edges, preserve flat areas)
      6. CLAHE + saturation      (final contrast & color polish)
    """
    # 1. Color-aware denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None,
        h=8, hColor=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # 2. MSRCR
    retinex = _msrcr(denoised, sigmas=(10, 60, 200), restore_factor=140.0)

    # 3. Percentile stretch — use the full 0-255 range
    stretched = _percentile_stretch(retinex, lo=0.5, hi=99.5)

    # 4. FFT high-boost sharpening
    freq_sharp = _freq_sharpen(stretched, strength=1.8)

    # 5. Adaptive edge sharpening
    sharp = _adaptive_sharpen(freq_sharp, strength=1.4)

    # 6. CLAHE on L + saturation boost in LAB
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    a = np.clip((a.astype(np.float32) - 128) * 1.25 + 128, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * 1.25 + 128, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


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

    # 1. Color-aware denoising - tam kalite parametreler (PC icin yeterli CPU/RAM var)
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None,
        h=5, hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # 2. CLAHE on L only - renk bozulmasi olmaz
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Yumusak unsharp mask - blurlu yuzlerde kenar/kas hatlarini ortaya cikarir
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.4, blur, -0.4, 0)

    # 4. AI super-resolution - dusuk cozunurluklu yuz icin en kritik adim
    try:
        return _sr_upscale(sharpened, scale=sr_scale)
    except Exception as e:
        print(f"[face_enhance] DNN SR basarisiz, INTER_CUBIC kullaniliyor: {e}")
        h, w = sharpened.shape[:2]
        return cv2.resize(sharpened, (w * sr_scale, h * sr_scale),
                          interpolation=cv2.INTER_CUBIC)


# --- Örnek kullanım ---
if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    if img is None:
        print("test.jpg bulunamadı, örnek görüntü oluşturuluyor...")
        img = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)

    result_clahe = process_l_channel(img, enhance_fn=clahe_enhance)
    result_gamma = process_l_channel(img, enhance_fn=lambda l: gamma_enhance(l, gamma=1.3))
    result_histeq = process_l_channel(img, enhance_fn=histogram_equalize)

    cv2.imwrite("output_clahe.jpg", result_clahe)
    cv2.imwrite("output_gamma.jpg", result_gamma)
    cv2.imwrite("output_histeq.jpg", result_histeq)
    print("Çıktılar kaydedildi.")
