"""Quick camera scanner — finds which camera indices are available.

Run this to find the correct --cam number for DroidCam.

Usage:
    python scripts/find_camera.py
"""

import cv2
import sys

print("=" * 50)
print("  CAMERA SCANNER")
print("  Finding all available cameras...")
print("=" * 50)
print()

found = []

for idx in range(10):
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        status = "OK - frame captured" if ret else "Opens but no frame"
        print(f"  Camera {idx}: {w}x{h}  [{status}]")
        if ret:
            found.append(idx)
        cap.release()
    else:
        cap.release()

print()
if found:
    print(f"Working cameras found: {found}")
    print()
    for idx in found:
        print(f"  To use camera {idx}:  python main.py --auto --cam {idx}")
else:
    print("No working cameras found!")
    print()
    print("Troubleshooting:")
    print("  1. Open the DroidCam Client app on your PC")
    print("  2. Open DroidCam on your phone")
    print("  3. Connect them (WiFi or USB)")
    print("  4. Run this script again")
    print()
    print("  If using a USB webcam, make sure it's plugged in.")
