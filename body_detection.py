import cv2
import os
import time
import winsound

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

cap = cv2.VideoCapture(0)

# Create folder for screenshots
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Video Writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

recording = False
screenshot_taken = False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    bodies = upperbody_cascade.detectMultiScale(gray, 1.1, 4)

    count = len(faces) + len(bodies)

    # Draw face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Draw upper body rectangles
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, f'People Detected: {count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 🔔 Sound Alarm if more than 3 people
    if count > 3:
        winsound.Beep(1000, 500)

    # 📸 Save Screenshot
    if count > 0 and not screenshot_taken:
        filename = f"screenshots/detected_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print("Screenshot Saved!")
        screenshot_taken = True

    if count == 0:
        screenshot_taken = False

    # 🎥 Start Recording When Person Detected
    if count > 0 and not recording:
        out = cv2.VideoWriter(f"record_{int(time.time())}.avi",
                              fourcc, 20.0,
                              (frame.shape[1], frame.shape[0]))
        recording = True
        print("Recording Started!")

    # Stop recording when no one detected
    if count == 0 and recording:
        out.release()
        recording = False
        print("Recording Stopped!")

    if recording:
        out.write(frame)

    cv2.imshow("Advanced CCTV Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()