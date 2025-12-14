
import os
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV is required. Please install: pip install opencv-python") from e

# -----------------------------
# Optional YAML support
# -----------------------------
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Loads YAML config from config/config.yaml.
    Requires PyYAML. If not installed, raises a clear instruction.
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to read config/config.yaml. Install: pip install pyyaml"
        ) from e

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files.sort()
    return files


def imread_bgr_uint8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def to_binary01(img: np.ndarray) -> np.ndarray:
    """
    Converts an image to binary {0,1}.
    Accepts grayscale or color images; returns 2D binary.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold to {0,255}
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bw > 0).astype(np.uint8)


def save_binary01_as_png(path: Path, binary01: np.ndarray) -> None:
    """
    Saves a binary {0,1} image as viewable PNG {0,255}.
    """
    view = (binary01.astype(np.uint8) * 255)
    cv2.imwrite(str(path), view)


# -----------------------------
# XOR bit-plane embed/extract helpers
# (These are included here as "system glue" to guarantee XOR consistency)
# -----------------------------
def extract_original_bitplane_bits(
    image_bgr: np.ndarray, bit_plane: int, channel: int, length: int
) -> np.ndarray:
    """
    Extracts the original bits from selected channel & bit_plane as a 1D array.
    Order: row-major, from top-left.
    """
    ch = image_bgr[:, :, channel]
    bits = ((ch >> bit_plane) & 1).astype(np.uint8).flatten()
    if length > bits.size:
        raise ValueError(f"Requested length={length} exceeds capacity={bits.size}.")
    return bits[:length]


def xor_bitstream(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Bitstream shape mismatch: {a.shape} vs {b.shape}")
    return (a ^ b).astype(np.uint8)


# -----------------------------
# Import project modules (must match your team contracts)
# -----------------------------
def import_modules():
    """
    Imports all modules with clear error messages if contracts are missing.
    """
    try:
        # A: watermark design + encoding
        from watermark_design.design import design_watermark
        from watermark_design.encode import encode_to_bitstream

        # B: embedding
        from embedding.embed import embed_watermark

        # C: extraction + decoding
        from extraction.extract import extract_bitstream
        from extraction.decode import decode_to_image

        # D: attacks + evaluation
        from attacks.jpeg_attack import simulate_jpeg_attack
        from attacks.noise_attack import simulate_noise_attack

        from evaluation.metrics import calculate_psnr, calculate_ber

    except Exception as e:
        raise ImportError(
            "Failed to import project modules. Ensure your project structure matches:\n"
            "src/watermark_design/{design.py, encode.py}\n"
            "src/embedding/embed.py\n"
            "src/extraction/{extract.py, decode.py}\n"
            "src/attacks/{jpeg_attack.py, noise_attack.py}\n"
            "src/evaluation/metrics.py\n"
            "And each file defines the required functions.\n"
            f"Original error: {repr(e)}"
        ) from e

    return {
        "design_watermark": design_watermark,
        "encode_to_bitstream": encode_to_bitstream,
        "embed_watermark": embed_watermark,
        "extract_bitstream": extract_bitstream,
        "decode_to_image": decode_to_image,
        "simulate_jpeg_attack": simulate_jpeg_attack,
        "simulate_noise_attack": simulate_noise_attack,
        "calculate_psnr": calculate_psnr,
        "calculate_ber": calculate_ber,
    }


# -----------------------------
# Main experiment pipeline
# -----------------------------
@dataclass
class ExperimentCase:
    cover_path: Path
    attack_type: str
    attack_param: str
    psnr: float
    ber: float
    stego_path: Path
    attacked_path: Path
    extracted_path: Path


def main():
    root = Path(__file__).resolve().parents[1]  # project root (digital-watermarking-project/)
    config_path = root / "config" / "config.yaml"

    cfg = load_config(config_path)

    # Read config
    w_h = int(cfg["watermark"]["height"])
    w_w = int(cfg["watermark"]["width"])
    bit_plane = int(cfg["embedding"]["bit_plane"])
    channel = int(cfg["embedding"]["channel"])

    jpeg_qualities = cfg.get("attacks", {}).get("jpeg_quality", [95, 85, 75])
    noise_sigmas = cfg.get("attacks", {}).get("gaussian_noise_sigma", [2, 5, 10])

    psnr_threshold = float(cfg.get("evaluation", {}).get("psnr_threshold", 40.0))

    # Paths
    cover_dir = root / "data" / "cover"
    watermark_dir = root / "data" / "watermark"
    logo_path = watermark_dir / "logo.png"

    out_root = root / "output"
    out_stego = out_root / "stego"
    out_attacked = out_root / "attacked"
    out_extracted = out_root / "extracted"
    ensure_dir(out_stego)
    ensure_dir(out_attacked)
    ensure_dir(out_extracted)

    # Import modules
    mods = import_modules()

    # Load / build watermark
    if logo_path.exists():
        logo_raw = imread_bgr_uint8(logo_path)
        watermark01 = to_binary01(logo_raw)
        # Resize to config size if needed (nearest keeps it binary)
        if watermark01.shape != (w_h, w_w):
            watermark01 = cv2.resize(watermark01, (w_w, w_h), interpolation=cv2.INTER_NEAREST)
            watermark01 = (watermark01 > 0).astype(np.uint8)
    else:
        watermark01 = mods["design_watermark"](size=(w_h, w_w)).astype(np.uint8)
        watermark01 = (watermark01 > 0).astype(np.uint8)
        # Save generated watermark for reproducibility
        ensure_dir(watermark_dir)
        save_binary01_as_png(logo_path, watermark01)

    watermark_bits = mods["encode_to_bitstream"](watermark01).astype(np.uint8)
    watermark_len = watermark_bits.size

    # List cover images
    covers = list_images(cover_dir)
    if not covers:
        raise FileNotFoundError(
            f"No cover images found in {cover_dir}. Put PNG/BMP/JPG images into data/cover/."
        )

    results: List[ExperimentCase] = []

    print("=== Experiment Config ===")
    print(f"Watermark size: {w_h}x{w_w} (len={watermark_len})")
    print(f"Embedding: XOR bit-plane | bit_plane={bit_plane} | channel(BGR)={channel}")
    print(f"JPEG qualities: {jpeg_qualities}")
    print(f"Noise sigmas: {noise_sigmas}")
    print("=========================")

    for cover_path in covers:
        cover = imread_bgr_uint8(cover_path)
        H, W, C = cover.shape
        capacity = H * W
        if watermark_len > capacity:
            raise ValueError(
                f"Watermark too large for cover image {cover_path.name}. "
                f"Need <= {capacity} bits but got {watermark_len}. "
                f"Reduce watermark size or embed across multiple channels/planes."
            )

        base_name = cover_path.stem

        # 1) XOR-based embedding requires original bitstream of the target plane
        original_bits = extract_original_bitplane_bits(cover, bit_plane, channel, watermark_len)
        embedded_bits = xor_bitstream(original_bits, watermark_bits)

        # 2) Create stego image by embedding "embedded_bits" into bit_plane/channel
        stego = mods["embed_watermark"](cover, embedded_bits, bit_plane, channel)

        stego_path = out_stego / f"{base_name}_stego.png"
        cv2.imwrite(str(stego_path), stego)

        # 3) Evaluate imperceptibility (PSNR between cover and stego)
        psnr_clean = float(mods["calculate_psnr"](cover, stego))
        if psnr_clean < psnr_threshold:
            print(
                f"[WARN] PSNR below threshold ({psnr_clean:.2f} dB < {psnr_threshold} dB) "
                f"for {cover_path.name}. Consider using a lower bit_plane."
            )

        # 4) Extract watermark from clean stego (no attack)
        extracted_bits_clean = mods["extract_bitstream"](stego, bit_plane, channel, watermark_len).astype(np.uint8)

        # XOR inversion: watermark = original_bit XOR embedded_bit
        recovered_w_bits_clean = xor_bitstream(original_bits, extracted_bits_clean)

        recovered_w_clean = mods["decode_to_image"](recovered_w_bits_clean, (w_h, w_w)).astype(np.uint8)
        recovered_w_clean = (recovered_w_clean > 0).astype(np.uint8)

        extracted_path_clean = out_extracted / f"{base_name}_none_logo.png"
        save_binary01_as_png(extracted_path_clean, recovered_w_clean)

        ber_clean = float(mods["calculate_ber"](watermark_bits, recovered_w_bits_clean))
        results.append(
            ExperimentCase(
                cover_path=cover_path,
                attack_type="none",
                attack_param="-",
                psnr=psnr_clean,
                ber=ber_clean,
                stego_path=stego_path,
                attacked_path=stego_path,
                extracted_path=extracted_path_clean,
            )
        )

        # 5) Attacks loop
        # 5.1 JPEG
        for q in jpeg_qualities:
            attacked = mods["simulate_jpeg_attack"](stego, int(q))
            attacked_path = out_attacked / f"{base_name}_jpeg_{int(q)}.jpg"
            # simulate_jpeg_attack may already return decoded image, but we save for record
            cv2.imwrite(str(attacked_path), attacked)

            # Extract & recover
            extracted_bits = mods["extract_bitstream"](attacked, bit_plane, channel, watermark_len).astype(np.uint8)
            recovered_w_bits = xor_bitstream(original_bits, extracted_bits)
            recovered_w = mods["decode_to_image"](recovered_w_bits, (w_h, w_w)).astype(np.uint8)
            recovered_w = (recovered_w > 0).astype(np.uint8)

            extracted_path = out_extracted / f"{base_name}_jpeg_{int(q)}_logo.png"
            save_binary01_as_png(extracted_path, recovered_w)

            psnr_val = float(mods["calculate_psnr"](cover, attacked))
            ber_val = float(mods["calculate_ber"](watermark_bits, recovered_w_bits))

            results.append(
                ExperimentCase(
                    cover_path=cover_path,
                    attack_type="jpeg",
                    attack_param=f"Q={int(q)}",
                    psnr=psnr_val,
                    ber=ber_val,
                    stego_path=stego_path,
                    attacked_path=attacked_path,
                    extracted_path=extracted_path,
                )
            )

        # 5.2 Noise
        for sigma in noise_sigmas:
            attacked = mods["simulate_noise_attack"](stego, float(sigma))
            attacked_path = out_attacked / f"{base_name}_noise_{float(sigma)}.png"
            cv2.imwrite(str(attacked_path), attacked)

            extracted_bits = mods["extract_bitstream"](attacked, bit_plane, channel, watermark_len).astype(np.uint8)
            recovered_w_bits = xor_bitstream(original_bits, extracted_bits)
            recovered_w = mods["decode_to_image"](recovered_w_bits, (w_h, w_w)).astype(np.uint8)
            recovered_w = (recovered_w > 0).astype(np.uint8)

            extracted_path = out_extracted / f"{base_name}_noise_{float(sigma)}_logo.png"
            save_binary01_as_png(extracted_path, recovered_w)

            psnr_val = float(mods["calculate_psnr"](cover, attacked))
            ber_val = float(mods["calculate_ber"](watermark_bits, recovered_w_bits))

            results.append(
                ExperimentCase(
                    cover_path=cover_path,
                    attack_type="noise",
                    attack_param=f"sigma={float(sigma)}",
                    psnr=psnr_val,
                    ber=ber_val,
                    stego_path=stego_path,
                    attacked_path=attacked_path,
                    extracted_path=extracted_path,
                )
            )

    # 6) Write CSV summary
    csv_path = out_root / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cover_image",
            "attack_type",
            "attack_param",
            "psnr_db",
            "ber",
            "stego_path",
            "attacked_path",
            "extracted_logo_path"
        ])
        for r in results:
            writer.writerow([
                r.cover_path.name,
                r.attack_type,
                r.attack_param,
                f"{r.psnr:.4f}",
                f"{r.ber:.6f}",
                str(r.stego_path.relative_to(root)),
                str(r.attacked_path.relative_to(root)),
                str(r.extracted_path.relative_to(root)),
            ])

    print("\n=== DONE ===")
    print(f"Results CSV: {csv_path}")
    print(f"Stego images: {out_stego}")
    print(f"Attacked images: {out_attacked}")
    print(f"Extracted logos: {out_extracted}")
    print("============")


if __name__ == "__main__":
    """
    Run from project root:
        python src/main.py

    Notes:
    - This main.py assumes XOR-based embedding consistency:
        embedded_bit = original_bit XOR watermark_bit
        recovered_watermark = original_bit XOR extracted_embedded_bit
    - Ensure your embed_watermark() performs bit-plane replacement on the provided bitstream.
    """
    main()
