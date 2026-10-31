
import cv2
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

CHIP_DIR = ROOT / "data/image_validation/potato_chips"
NON_CHIP_DIR = ROOT / "data/image_validation/not_potato_chips"

CHIP_DIR.mkdir(parents=True, exist_ok=True)
NON_CHIP_DIR.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit(1)

print("\n========== POTATOGUARD DATA COLLECTION ==========")
print("P = Capture potato-chip image")
print("N = Capture non-potato-chip image")
print("Q = Quit")
print("=================================================\n")

chip_count = 0
non_chip_count = 0

while True:
    success, frame = camera.read()

    if not success:
        print("Camera frame could not be captured.")
        break

    display = frame.copy()

    cv2.putText(
        display,
        "P: Potato Chips | N: Other | Q: Quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("PotatoGuard - Dataset Collection", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):
        chip_count += 1
        filename = f"camera_chip_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(CHIP_DIR / filename), frame)
        print("Saved potato-chip image:", filename)

    elif key == ord("n"):
        non_chip_count += 1
        filename = f"camera_other_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(NON_CHIP_DIR / filename), frame)
        print("Saved non-chip image:", filename)

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("\nCollection completed.")
print("New potato-chip images:", chip_count)
print("New non-chip images:", non_chip_count)
