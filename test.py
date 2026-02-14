import cv2
import numpy as np

# Load image
img = cv2.imread("road.jpg")
height = img.shape[0]
width = img.shape[1]

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 30, 100)

# Create mask (triangle region of interest)
mask = np.zeros_like(edges)

polygon = np.array([[
    (0, height),
    (width, height),
    (width//2, height//2)
]], np.int32)

cv2.fillPoly(mask, polygon, 255)

masked_edges = cv2.bitwise_and(edges, mask)

# Hough Transform
lines = cv2.HoughLinesP(
    masked_edges,
    1,
    np.pi/180,
    50,
    minLineLength=40,
    maxLineGap=100
)

# Draw filtered lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Remove almost horizontal lines
        if abs(slope) < 0.5:
            continue

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("Clean Lane Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
