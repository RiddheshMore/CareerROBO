import cv2
import dlib
import face_recognition

# Initialize Dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Known faces with their images
KNOWN_FACES = {
    #"Riddhesh": "2025-01-08-034449.jpg",  # Replace with the actual path to Riddhesh's image
    #"Shubham": "Media/shubham.jpg"     # Replace with the actual path to Shubham's image
}

# Function to load known face encodings
def load_known_faces():
    known_encodings = {}
    for name, image_path in KNOWN_FACES.items():
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings[name] = encoding
    return known_encodings

# Main function for face detection and recognition
def face_detection_and_recognition(known_encodings):
    # Start video capture from webcam
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Convert frame to grayscale and RGB
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            detections, _, _ = detector.run(gray, 1)

            # Recognize faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(list(known_encodings.values()), face_encoding)
                name = "Unknown"
                color = (0, 0, 255)  # Default color for "Unknown" (red)

                if True in matches:
                    match_index = matches.index(True)
                    name = list(known_encodings.keys())[match_index]
                    color = (0, 255, 0)  # Green for recognized individuals

                # Draw rectangle and name around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display recognition status
            if len(face_encodings) > 0 and name == "Unknown":
                cv2.putText(frame, "Face Not Recognized", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(face_encodings) > 0:
                cv2.putText(frame, "Face Successfully Recognized", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow('Face Detection and Recognition', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    # Load known face encodings
    known_face_encodings = load_known_faces()

    # Run the face detection and recognition
    face_detection_and_recognition(known_face_encodings)
