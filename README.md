# 🧾 Soufueled Task 2 – Nail Detection 

## 🎯 Objective:
Detect and count the number of **non-rusted (silver)** nails falling in a 10-second video.

---

## ✅ Approach

### 1. Video Input & Preprocessing
- Loaded the input video using OpenCV.
- Skipped the first 1 second (~30 frames) to avoid early detection of rusted nails.
- Resized each frame for consistent processing.

### 2. Color-Based Filtering
- Converted each frame to **HSV color space**.
- Applied a **tight HSV mask** to detect only **silver (non-rusted)** nails.
- Excluded rusted nails (brown) by filtering out high saturation and low brightness regions.

### 3. Motion Detection
- Used **Background Subtraction (MOG2)** to isolate falling/moving objects.
- Combined it with the color mask to focus only on **falling silver nails**.

### 4. Contour Detection & Tracking
- Extracted contours and filtered by area to remove noise.
- Used **centroid-based tracking** to count each unique nail only once, avoiding double counting.

### 5. Slowed Video Playback
- Slowed video playback by **5×** (`cv2.waitKey(150)`) to ensure accurate detection and easier manual verification.

---

## 📈 Outcome
- Successfully detected **25 non-rusted (silver) nails** from the given video.
- Output video saved as:  
  🎥 `soufueled_task2_khushi.mp4`

---

## 🤔 Why Not Use TensorFlow or Transformer-Based Models?
Since this task involves a **single short video** with a clearly defined visual pattern, using OpenCV with classical techniques is more efficient and interpretable.  
Deep learning models like CNNs or transformers are typically better suited for **large datasets** and require time for training and tuning, which is unnecessary for this limited-scope task.

---

## 📦 Output File
- 🎥 `soufueled_task2_khushi.mp4`: Final video showing slowed-down real-time detection and counting of silver nails.
- 🔗 [View Output Video on Google Drive](https://drive.google.com/file/d/13wP8JHzTpmIKwVnzL3e1zmOk5t5wqzG_/view?usp=sharing)
