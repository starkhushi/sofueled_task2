import cv2
import numpy as np
import imutils

video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

fgbg = cv2.createBackgroundSubtractorMOG2()

detected_centroids = []
silver_nail_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip first 1 second (assuming 30 FPS)
    if frame_count < 30:
        continue

    # Resize for consistency
    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Background subtraction
    fgmask = fgbg.apply(frame)

    # Updated color range for silver (non-rusted) nails
    lower_silver = np.array([0, 0, 160])
    upper_silver = np.array([180, 40, 255])
    silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)

    # Combine motion and color masks
    combined_mask = cv2.bitwise_and(fgmask, silver_mask)

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 800:  # Adjust for nail size
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            # Check for duplicates
            duplicate = False
            for prev_cx, prev_cy in detected_centroids:
                if abs(cx - prev_cx) < 20 and abs(cy - prev_cy) < 20:
                    duplicate = True
                    break

            if not duplicate:
                detected_centroids.append((cx, cy))
                silver_nail_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Nail {silver_nail_count}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f'Silver Nails Counted: {silver_nail_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame (slowed down by 5x)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(150) & 0xFF == ord('q'):  # Slower display
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Total silver (non-rusted) nails counted: {silver_nail_count}")
