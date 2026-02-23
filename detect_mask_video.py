import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("model/mask_detector.keras")

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Labels and colors
labels_dict = {0: 'No Mask', 1: 'Mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

print("[INFO] Starting webcam...")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        result = model.predict(face_img)
        label = np.argmax(result, axis=1)[0]

        color = color_dict[label]
        label_text = labels_dict[label]

       
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam stopped.")
