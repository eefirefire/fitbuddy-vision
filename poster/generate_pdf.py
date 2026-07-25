"""
Generates a single-page A0 PDF from fitbuddy_poster.html.
Uses Chrome screenshot at full A0 pixel size, then img2pdf to embed at A0 dimensions.
Run from the poster/ folder with the local server already running on port 8765.
"""
import subprocess, os, time
from PIL import Image
import img2pdf

CHROME  = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL     = "http://localhost:8765/fitbuddy_poster.html?fullsize=1"
OUT     = os.path.dirname(os.path.abspath(__file__))
TMP_PNG = os.path.join(OUT, "_tmp_a0.png")
PDF_OUT = os.path.join(OUT, "fitbuddy_poster.pdf")
PNG_OUT = os.path.join(OUT, "fitbuddy_poster.png")

# A0 at 96 dpi  (841mm / 25.4 * 96 = 3179, 1189mm / 25.4 * 96 = 4494)
W, H = 3179, 4494

print(f"Taking full-size A0 screenshot ({W}×{H}px)…")
subprocess.run([
    CHROME,
    "--headless=new", "--disable-gpu", "--no-sandbox",
    "--disable-web-security",          # allow Google Fonts from localhost
    "--hide-scrollbars",
    f"--screenshot={TMP_PNG}",
    f"--window-size={W},{H+20}",       # +20 to avoid browser chrome clipping
    "--force-device-scale-factor=1",
    "--virtual-time-budget=6000",
    URL,
], check=True, timeout=90)

time.sleep(1)

if not os.path.exists(TMP_PNG):
    raise FileNotFoundError("Chrome did not produce a screenshot.")

# Crop to exact A0 pixel dimensions
img = Image.open(TMP_PNG)
print(f"  Screenshot size: {img.size}")
img_cropped = img.crop((0, 0, W, min(H, img.height)))
img_cropped.save(TMP_PNG)
img_cropped.save(PNG_OUT)       # also save as the poster PNG
print(f"  Cropped to: {img_cropped.size}")

# Convert to A0 PDF with img2pdf
a0_pt = (img2pdf.mm_to_pt(841), img2pdf.mm_to_pt(1189))
layout = img2pdf.get_layout_fun(a0_pt)
with open(PDF_OUT, "wb") as f:
    f.write(img2pdf.convert(TMP_PNG, layout_fun=layout))

os.remove(TMP_PNG)
print(f"\nDone!")
print(f"  PDF : {PDF_OUT}")
print(f"  PNG : {PNG_OUT}")
