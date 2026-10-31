
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10

DEST = Path("data/image_validation/not_potato_chips")
DEST.mkdir(parents=True, exist_ok=True)

print("Downloading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Combine training and test images
images = np.concatenate([x_train, x_test], axis=0)

# Select 1,500 images randomly and reproducibly
rng = np.random.default_rng(42)
indices = rng.choice(len(images), size=1500, replace=False)

saved = 0

for index in indices:
    image = images[index]

    # Resize for the image validator
    image = cv2.resize(image, (224, 224))

    output_path = DEST / f"negative_{saved + 1:04d}.jpg"

    if cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        saved += 1

print(f"\nDownload and extraction completed.")
print(f"Negative images saved: {saved}")
print(f"Folder: {DEST.resolve()}")
