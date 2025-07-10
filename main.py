import cv2
import numpy as np
import sys
import os

# Configuration
THRESHOLD_PADDING = 15      # Lower = more sensitive
MIN_CLUSTER_SIZE = 6        # Ignore very small specks
BORDER_CROP = 0.03          # Ignore outer X% of image (vignette area)

def detect_spots(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop borders to avoid vignetting influence
    h, w = gray.shape
    crop_x = int(w * BORDER_CROP)
    crop_y = int(h * BORDER_CROP)
    roi = gray[crop_y:h-crop_y, crop_x:w-crop_x]

    # Optional: Contrast enhancement (CLAHE works well on low-contrast areas)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi = clahe.apply(roi)

    mean = np.mean(roi)
    lower = max(0, mean - THRESHOLD_PADDING)
    upper = min(255, mean + THRESHOLD_PADDING)

    mask = cv2.inRange(roi, 0, lower) | cv2.inRange(roi, upper, 255)

    # Pad mask back to original image size
    full_mask = np.zeros_like(gray, dtype=np.uint8)
    full_mask[crop_y:h-crop_y, crop_x:w-crop_x] = mask

    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(full_mask)
    spots = []
    for i in range(1, num_labels):
        size = stats[i, cv2.CC_STAT_AREA]
        if size >= MIN_CLUSTER_SIZE:
            x, y = int(centroids[i][0]), int(centroids[i][1])
            spots.append((x, y, size))
    return spots

def annotate_image(image, spots):
    output = image.copy()
    h, w = output.shape[:2]
    font_scale = h / 1000.0
    font_thickness = max(1, int(h / 600))

    for x, y, size in spots:
        radius = int(np.sqrt(size))
        cv2.circle(output, (x, y), radius + 3, (0, 0, 255), 2)

    text = f"{len(spots)} spot{'s' if len(spots) != 1 else ''} found"
    cv2.putText(output, text, (int(w * 0.02), int(h * 0.05)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
    return output

def main(image_path):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        print(f"Error: Could not read image: {image_path}")
        return

    spots = detect_spots(image)
    annotated = annotate_image(image, spots)

    base = os.path.splitext(image_path)[0]
    output_path = base + "_StaubSpion.jpg"
    cv2.imwrite(output_path, annotated)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Drag a JPG file onto this .exe to run detection.")
    else:
        for image_path in sys.argv[1:]:
            main(image_path)
