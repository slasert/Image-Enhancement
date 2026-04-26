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
    enh_gray  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

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


def print_metrics(metrics: dict, label: str = "Sonuc") -> None:
    """Metrikleri okunakli bicimde yazdirir."""
    print(f"\n--- {label} ---")
    print(f"  PSNR  : {metrics['psnr_db']} dB")
    print(f"  SSIM  : {metrics['ssim']}")
    print(f"  PSNR per channel -> "
          f"B: {metrics['psnr_per_channel']['B']} | "
          f"G: {metrics['psnr_per_channel']['G']} | "
          f"R: {metrics['psnr_per_channel']['R']}")


# --- Ornek kullanim ---
if __name__ == "__main__":
    from lab_processing import process_l_channel, clahe_enhance, bilateral_enhance

    img = cv2.imread("test.jpg")
    if img is None:
        print("test.jpg bulunamadi, ornek goruntu olusturuluyor...")
        img = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)

    result_clahe     = process_l_channel(img, enhance_fn=clahe_enhance)
    result_bilateral = process_l_channel(img, enhance_fn=bilateral_enhance)

    print_metrics(compute_metrics(img, result_clahe),     label="CLAHE")
    print_metrics(compute_metrics(img, result_bilateral), label="Bilateral Filter")
