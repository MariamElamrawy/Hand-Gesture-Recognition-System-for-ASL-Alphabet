import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time
import os

# Open the first webcam (usually 0)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Images/A"
counter = 0
os.makedirs(folder, exist_ok=True)

imgWhite = None

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'S' to save | Press 'Q' to quit")

while True:
    # Read frame
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Get image dimensions
        img_h, img_w, _ = img.shape
        
        # Calculate crop boundaries with safety checks
        y1 = max(0, y - offset)
        y2 = min(img_h, y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img_w, x + w + offset)
        
        imgCrop = img[y1:y2, x1:x2]

        # Only process if crop is valid
        if imgCrop.size > 0 and imgCrop.shape[0] > 10 and imgCrop.shape[1] > 10:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            h_crop, w_crop, _ = imgCrop.shape
            aspectRatio = h_crop / w_crop

            if aspectRatio > 1:
                # Height is bigger
                k = imgSize / h_crop
                wCal = int(k * w_crop)
                if wCal > 0 and wCal <= imgSize:
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = (imgSize - wCal) // 2
                    imgWhite[:, wGap:wGap + wCal] = imgResize
            else: 
                # Width is bigger
                k = imgSize / w_crop
                hCal = int(k * h_crop)
                if hCal > 0 and hCal <= imgSize:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = (imgSize - hCal) // 2
                    imgWhite[hGap:hGap + hCal, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Show the frame
    cv2.imshow("Webcam Feed", img)

    key = cv2.waitKey(1) & 0xFF

    # Exit on 'q' key
    if key == ord('q'):
        break

    # Save on 's' key
    if key == ord("s"):
        if imgWhite is not None:
            counter += 1
            cv2.imwrite(f'{folder}/Image_{counter}_{int(time.time())}.jpg', imgWhite)
            print(f"Saved image {counter}")
        else:
            print("No hand detected")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
print(f"\nTotal saved: {counter} images")