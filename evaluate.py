import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob
from tabulate import tabulate
from app.restoration import ImageRestorer

# Configuration
INPUT_DIR = "test_inputs"       # Put raw images here
OUTPUT_DIR = "test_results"     # The script will save its own results here

def run_evaluation():
    # 1. Initialize the AI Brain once
    print("--- 🧠 Loading AI Model for Testing ---")
    try:
        restorer = ImageRestorer()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Prepare Folders
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    if not files:
        print(f"❌ No images found in '{INPUT_DIR}'. Please add some test images.")
        return

    report_data = []
    print(f"--- 🧪 Starting Test on {len(files)} images ---")

    for f in files:
        filename = os.path.basename(f)
        print(f"Processing: {filename}...")
        
        # Read Original
        img_orig = cv2.imread(f)
        if img_orig is None: continue

        # Run AI Inference
        # We disable face_enhance for pure metric testing, or enable it if that's your focus
        # Usually metrics are better tested without face paste-back artifacts, but let's keep it True.
        img_rest = restorer.process_image(img_orig, face_enhance=True)
        
        if img_rest is None:
            print("Failed to restore.")
            continue

        # Save the result so you can look at it later
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, img_rest)

        # --- CALCULATE METRICS ---
        h_orig, w_orig, _ = img_orig.shape
        h_rest, w_rest, _ = img_rest.shape
        
        # Resize original to match result for mathematical comparison
        img_orig_resized = cv2.resize(img_orig, (w_rest, h_rest), interpolation=cv2.INTER_CUBIC)

        # PSNR (Noise Removal Score)
        current_psnr = psnr(img_orig_resized, img_rest)

        # SSIM (Structure/Sharpness Score)
        gray_orig = cv2.cvtColor(img_orig_resized, cv2.COLOR_BGR2GRAY)
        gray_rest = cv2.cvtColor(img_rest, cv2.COLOR_BGR2GRAY)
        current_ssim = ssim(gray_orig, gray_rest)

        # File Size Growth
        size_orig_kb = os.path.getsize(f) / 1024
        size_rest_kb = os.path.getsize(save_path) / 1024

        report_data.append([
            filename,
            f"{w_orig}x{h_orig} -> {w_rest}x{h_rest}",
            f"{size_orig_kb:.1f} -> {size_rest_kb:.1f} KB",
            f"{current_psnr:.2f}",
            f"{current_ssim:.4f}"
        ])

    # 3. Print Professional Report
    headers = ["Filename", "Dims (Old->New)", "Size (Old->New)", "PSNR (Noise)", "SSIM (Sharpness)"]
    print("\n" + "="*80)
    print("FINAL QUALITY ASSURANCE REPORT")
    print("="*80)
    print(tabulate(report_data, headers=headers, tablefmt="grid"))
    print("\nNOTE: ")
    print("- PSNR > 30 is generally considered excellent quality.")
    print("- SSIM > 0.8 means structure is very well preserved.")
    print(f"- Visual results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_evaluation()