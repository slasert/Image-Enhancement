import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """
    Original ve iyilestirilmis goruntu arasindaki SSIM ve PSNR degerlerini hesaplar.

    Args:
        original: Orijinal BGR goruntu (uint8)
        enhanced: Islenmis BGR goruntu (uint8)

    Returns:
        {
            "psnr_db"  : float  — yuksek = daha iyi (tipik: 30-50 dB),
            "ssim"     : float  — 0.0 ile 1.0 arasi (1.0 = ayni goruntu),
            "psnr_per_channel": {"B": float, "G": float, "R": float}
        }
    """
    if original.shape != enhanced.shape:
        raise ValueError(
            f"Goruntu boyutlari eslesmiyor: {original.shape} vs {enhanced.shape}"
        )

    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    ssim_val = ssim(orig_gray, enh_gray, data_range=255)
    psnr_val = psnr(orig_gray, enh_gray, data_range=255)

    channel_names = ["B", "G", "R"]
    psnr_channels = {
        ch: psnr(original[:, :, i], enhanced[:, :, i], data_range=255)
        for i, ch in enumerate(channel_names)
    }

    return {
        "psnr_db": round(float(psnr_val), 4),
        "ssim":    round(float(ssim_val), 4),
        "psnr_per_channel": {k: round(v, 4) for k, v in psnr_channels.items()},
    }
