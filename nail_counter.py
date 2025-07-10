import cv2
import numpy as np
import imutils

video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

fgbg = cv2.createBackgroundSubtractorMOG2()

detected_centroids = []
falling_nail_centroids = []
total_nail_count = 0
falling_nail_count = 0
frame_count = 0

falling_start_frame = 10  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Optional: Apply CLAHE to V channel
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge((h, s, v))

    fgmask = fgbg.apply(frame)

    # Adjusted color range for silver (tune as needed)
    lower_silver = np.array([0, 0, 140])
    upper_silver = np.array([180, 60, 255])

    silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)
    combined_mask = cv2.bitwise_and(fgmask, silver_mask)

    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 < area < 1200:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            duplicate = False
            for prev_cx, prev_cy in detected_centroids:
                if abs(cx - prev_cx) < 18 and abs(cy - prev_cy) < 18:
                    duplicate = True
                    break

            if not duplicate:
                detected_centroids.append((cx, cy))
                total_nail_count += 1

                # Count as falling nail only if detected after 1 second
                if frame_count >= falling_start_frame:
                    falling_nail_centroids.append((cx, cy))
                    falling_nail_count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Nail {total_nail_count}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display both counts
    cv2.putText(frame, f'Total Nails: {total_nail_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Falling Nails: {falling_nail_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Total silver (non-rusted) nails counted: {total_nail_count}")
print(f"✅ Falling nails (non-rusted) counted after 0.5 second: {falling_nail_count}")
