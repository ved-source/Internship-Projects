Python 3.8+ installed and available on PATH.​

A working OpenCV GUI backend (cv2.imshow requires a desktop environment; not supported in headless servers without virtual display).​

Files required and their expected locations:

Haar cascade XML: C:\Users\saive\Downloads\haarcascade_frontalface_default.xml (or update the path in code).​

Keras model file: C:\Users\saive\Downloads\gender_detection.keras (or update the path in code).​

Input video: C:\Users\saive\Downloads\WhatsApp Video 2025-09-26 at 17.00.54_f84d1346.mp4 (or use webcam/index by changing the path).​

Install dependencies
Create and activate a virtual environment (optional but recommended).

Install required packages:

pip install opencv-python numpy keras tensorflow
These satisfy cv2, NumPy, and Keras/TensorFlow for loading and running the model.​

Configure paths (if needed)
Edit these lines in gender_detection.py if your files are in different locations:

face_classifier = cv2.CascadeClassifier(r"C:\Users\saive\Downloads\haarcascade_frontalface_default.xml").​

classifier = load_model(r"C:\Users\saive\Downloads\gender_detection.keras").​

cap = cv2.VideoCapture(r"C:\Users\saive\Downloads\WhatsApp Video 2025-09-26 at 17.00.54_f84d1346.mp4").​

Tips:

To use a webcam instead of a video file, change to cap = cv2.VideoCapture(0) for default camera index.​

Ensure the Haar cascade XML is the frontal-face variant and readable by OpenCV; cv2.CascadeClassifier returns empty if the path is wrong.​

Run the script
From the project directory:

python gender_detection.py

Controls:

A window named “Gender Detector” will open and display processed frames with rectangles and labels for detected faces.​

Press q in the window to exit gracefully; resources will be released and windows destroyed.​

How it works (quick read)
Reads frames in a loop from cv2.VideoCapture, converts to grayscale for face detection, and detects faces via detectMultiScale with scaleFactor=1.3 and minNeighbors=5.​

For each face, crops the color ROI, resizes to 96x96, normalizes to , converts to array, expands batch dimension, and runs classifier.predict to get logits/probabilities.​

Picks the argmax index and maps to gender_labels = ['Male', 'Female'], drawing the label above the face; on empty ROI, writes “No Face”.​

Common issues and fixes
No window or crash at imshow: ensure a local GUI environment; on Linux servers, set up X11 or use a virtual display (e.g., xvfb), or modify code to write frames to a video file instead of showing.​

Empty face detections: verify the cascade path and that CascadeClassifier loaded correctly; print face_classifier.empty() to debug, adjust scaleFactor/minNeighbors and ensure faces are frontal in the video.​

Model load errors: confirm the .keras file exists and is compatible with the installed Keras/TensorFlow version; retrain or save the model with the same version used at inference if signatures mismatch.​

Slow inference: reduce input video resolution, or consider batching predictions per frame by stacking all detected ROIs into one array and calling predict once per frame to reduce overhead