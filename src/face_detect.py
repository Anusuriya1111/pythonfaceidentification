import os
import cv2
import sys

# Absolute path of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cascade file path
cascPath = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascPath)

if faceCascade.empty():
    raise IOError(f"Cannot load cascade file: {cascPath}")

# Image path from argument
imagePath = sys.argv[1]
image = cv2.imread(imagePath)
if image is None:
    raise IOError(f"Cannot load image file: {imagePath}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(f"Found {len(faces)} faces!")

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save output instead of imshow
output_path = os.path.join(BASE_DIR, "output.png")
cv2.imwrite(output_path, image)
print(f"Output saved to {output_path}")
