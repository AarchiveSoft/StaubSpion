import os
import sys

import cv2
import numpy as np

# ---------------- Configuration ----------------
THRESHOLD_PADDING = 20  # How much brighter/darker a pixel must be to be considered a "spot"
MIN_CLUSTER_SIZE = 3  # Minimum connected pixel count to be considered a real spot
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 2


# ------------------------------------------------

def detect_spots(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    lower = max(0, mean - THRESHOLD_PADDING)
    upper = min(255, mean + THRESHOLD_PADDING)

    mask = cv2.inRange(gray, 0, lower) | cv2.inRange(gray, upper, 255)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    spots = []

    for i in range(1, num_labels):
        size = stats[i, cv2.CC_STAT_AREA]
        if size >= MIN_CLUSTER_SIZE:
            x, y = int(centroids[i][0]), int(centroids[i][1])
            spots.append((x, y, size))

    return spots


def annotate_image(image, spots):
    output = image.copy()

    for x, y, size in spots:
        radius = int(np.sqrt(size))
        cv2.circle(output, (x, y), radius + 3, (0, 0, 255), 2)

    text = f"{len(spots)} spot{'s' if len(spots) != 1 else ''} found"
    cv2.putText(output, text, (10, 30), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    return output


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    spots = detect_spots(image)
    annotated = annotate_image(image, spots)

    # Save next to original file
    base, ext = os.path.splitext(image_path)
    output_path = f"StaubSpion_{base}"
    cv2.imwrite(output_path, annotated)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Drag a JPG file onto this .exe to run detection.")
    else:
        for image_path in sys.argv[1:]:
            main(image_path)
