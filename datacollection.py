import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Open the first webcam (usually 0)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255
    
        imgCrop = img[y - offset:y + h + offset , x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        imgWhite [0:imgCropShape[0],0 :imgCropShape[1]] = imgCrop

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Show the frame
    cv2.imshow("Webcam Feed", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

