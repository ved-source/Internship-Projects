import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(
    r"C:\Users\saive\Downloads\haarcascade_frontalface_default.xml"
)

# Load your Keras gender classification model
classifier = load_model(
    r"C:\Users\saive\Downloads\gender_detection.keras"
)

# Define labels
gender_labels = ['Male', 'Female']

# Start webcam
cap = cv2.VideoCapture(r"C:\Users\saive\Downloads\WhatsApp Video 2025-09-26 at 17.00.54_f84d1346.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection only
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract face from original color frame
        roi_color = frame[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color, (96, 96))  # Match model input

        if np.sum(roi_color) != 0:
            # Normalize and prepare for prediction
            roi = roi_color.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict gender
            prediction = classifier.predict(roi)[0]
            label = gender_labels[prediction.argmax()]

            # Show label above face
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show output frame
    cv2.imshow("Gender Detector", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
