import cv2
import dlib

# Initialize Dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Main function for face detection
def face_detection():
    # Start video capture from webcam
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Convert frame to grayscale as Dlib requires grayscale input
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            detections = detector(gray)

            if len(detections) > 0:
                # Draw rectangles around detected faces
                for detection in detections:
                    x, y, w, h = detection.left(), detection.top(), detection.width(), detection.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display "FACE DETECTED" message
                cv2.putText(frame, "FACE DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Display "FACE NOT DETECTED" message
                cv2.putText(frame, "FACE NOT DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show the frame
            cv2.imshow('Face Detection', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    # Run the face detection
    face_detection()
