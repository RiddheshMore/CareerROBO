import cv2
import dlib
import face_recognition
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import os
import time
from qibullet import SimulationManager
from qibullet import PepperVirtual
import gtts
import pygame
import threading
from concurrent.futures import ThreadPoolExecutor
from qibullet.camera import Camera
import requests
import threading
import webbrowser 
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr


# Function to communicate with Rasa and get response
def get_rasa_response(message):
    RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"  # Replace with your Rasa server URL
    payload = {"sender": "user", "message": message}
    response = requests.post(RASA_SERVER_URL, json=payload)
    return response.json()

# Function to make Pepper speak using TTS
def speak(message, filename):
    tts = gTTS(message)
    tts.save(filename)
    audio = AudioSegment.from_mp3(filename)
    play(audio)
    os.remove(filename)  # Optionally remove the temporary file after playing

# Functions for Pepper gestures
def wave(pepper):
    for _ in range(2):
        pepper.setAngles("RShoulderPitch", -0.5, 0.5)
        pepper.setAngles("RShoulderRoll", -1.5620, 0.5)
        pepper.setAngles("RElbowRoll", 1.5620, 0.5)
        time.sleep(1.0)
        pepper.setAngles("RElbowRoll", -1.5620, 0.5)
        time.sleep(1.0)

def head_nod(pepper):
    for _ in range(2):
        pepper.setAngles("HeadPitch", 0.5, 0.5)  # Nod down
        time.sleep(1.0)
        pepper.setAngles("HeadPitch", -0.5, 0.5)  # Nod up
        time.sleep(1.0)

# Function to open the HTML chat interface
def open_chat_html():
    webbrowser.open("chat.html")  # Replace with the path to your HTML file

# Async function to speak
def speak_async(message, filename):
    threading.Thread(target=speak, args=(message, filename)).start()

class FaceTracker:
    def __init__(self):
        self.history_size = 5  # Track last 5 frames
        self.detection_history = []  # Store detection history
        self.confidence_threshold = 0.6
        self.stability_threshold = 3  # Minimum consistent detections needed

    def update(self, detection: Optional[Dict]) -> Tuple[bool, Optional[str], float]:
        """
        Update detection history and return stable detection status
        Returns: (is_stable, name, confidence)
        """
        # Add new detection to history
        self.detection_history.append(detection)
        
        # Keep only last N frames
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
            
        # If no detections, return unstable
        if not self.detection_history or all(d is None for d in self.detection_history):
            return False, None, 0.0
            
        # Count detections of each name
        name_counts = {}
        for det in self.detection_history:
            if det is not None:
                name = det.get('name', 'Unknown')
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Find most common name
        if name_counts:
            most_common = max(name_counts.items(), key=lambda x: x[1])
            name, count = most_common
            
            # Calculate average confidence for this name
            confidences = [det.get('confidence', 0.0) 
                         for det in self.detection_history 
                         if det is not None and det.get('name') == name]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Check if detection is stable
            is_stable = (count >= self.stability_threshold and 
                        avg_confidence >= self.confidence_threshold)
            
            return is_stable, name, avg_confidence
        
        return False, None, 0.0


class PepperInteractiveSystem:
    def __init__(self, known_faces_dir: Optional[str] = None, use_webcam: bool = True):
        """
        Initialize the Pepper interactive system with face detection and behaviors.
        
        Args:
            known_faces_dir: Optional directory containing images of known faces
        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize webcam
        self.use_webcam = use_webcam
        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
            self.logger.info("Webcam initialized successfully")
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize simulation
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)
        
        # Initialize to default posture
        self.pepper.goToPosture("StandInit", 0.6)
        
        # Initialize face detection and tracking
        self.detector = dlib.get_frontal_face_detector()
        self.known_faces: Dict[str, np.ndarray] = {}
        self.face_tracker = FaceTracker()
        
        if not self.use_webcam:
            # Initialize Pepper's camera only if not using webcam
            self.camera_handle = self.pepper.subscribeCamera(
                PepperVirtual.ID_CAMERA_TOP,
                resolution=Camera.K_QVGA,
                fps=15.0
            )
            
            # Test Pepper's camera
            self.logger.info("Initializing Pepper's camera...")
            test_frame = self.pepper.getCameraFrame(self.camera_handle)
            if test_frame is not None:
                self.logger.info(f"Successfully got test frame with shape: {test_frame.shape}")
            else:
                self.logger.warning("Could not get test frame from Pepper's camera")
        
        # Initialize to default posture
        self.pepper.goToPosture("StandInit", 0.6)
        
        # Initialize face detection
        self.detector = dlib.get_frontal_face_detector()
        self.known_faces: Dict[str, np.ndarray] = {}
        
        # Initialize behavior control
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.running_tasks = {}
        
        # State tracking
        self.face_already_greeted = False
        self.last_greeting_time = 0
        self.greeting_cooldown = 10  # seconds between greetings
        
        # Load known faces if directory provided
        if known_faces_dir and os.path.exists(known_faces_dir):
            self.load_known_faces_from_directory(known_faces_dir)
    
    def process_camera_feed(self) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Process Pepper's camera feed for face detection.
        Uses the same approach as your working face_recognition.py script.
        """
        try:
            # Get frame from appropriate camera source
            if self.use_webcam:
                ret, img = self.cap.read()
                if not ret or img is None:
                    self.logger.warning("Failed to get webcam frame")
                    return None, []
            else:
                img = self.pepper.getCameraFrame(self.camera_handle)
                if img is None:
                    self.logger.warning("Failed to get Pepper camera frame")
                    return None, []
            
            # Convert formats
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            
            # face detection
            faces = self.detector(gray)
            detected_faces = []

            for face in faces:
                face_location = (face.top(), face.right(), face.bottom(), face.left())
                face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])
                
                name = "Unknown"
                confidence = 0.0
                
                if face_encoding and self.known_faces:
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        list(self.known_faces.values()),
                        face_encoding[0]
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        
                        if confidence >= self.face_tracker.confidence_threshold:
                            name = list(self.known_faces.keys())[best_match_index]
                
                detected_faces.append({
                    'name': name,
                    'location': face_location,
                    'confidence': confidence
                })
            
            return img, detected_faces
            
        except Exception as e:
            self.logger.error(f"Error processing camera feed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, []
            
    
    def load_known_faces_from_directory(self, directory: str) -> None:
        """Load all face images from a directory and create their encodings."""
        try:
            for filename in os.listdir(directory):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(filename)[0]
                    image_path = os.path.join(directory, filename)
                    self.add_known_face(name, image_path)
        except Exception as e:
            self.logger.error(f"Error loading faces from directory: {e}")

    def add_known_face(self, name: str, image_path: str) -> bool:
        """Add a single known face to the system."""
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                self.logger.warning(f"No face found in image for {name}")
                return False
                
            self.known_faces[name] = encodings[0]
            self.logger.info(f"Successfully added face encoding for {name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding known face {name}: {e}")
            return False

    def speak(self, text: str) -> None:
        """Make Pepper speak using text-to-speech."""
        try:
            tts = gtts.gTTS(text)
            filename = "speech.mp3"
            tts.save(filename)
            
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.quit()
            
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            self.logger.error(f"Error in speech synthesis: {e}")

    def wave(self) -> None:
        """Make Pepper perform a waving gesture."""
        try:
            for _ in range(3):
                self.pepper.setAngles("RShoulderPitch", -0.5, 0.5)
                self.pepper.setAngles("RShoulderRoll", -1.5620, 0.5)
                self.pepper.setAngles("RElbowRoll", 1.5620, 0.5)
                time.sleep(0.5)
                self.pepper.setAngles("RElbowRoll", -1.5620, 0.5)
                time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Error in wave gesture: {e}")

    def trigger_greeting(self, name: str) -> None:
        """Trigger Pepper's greeting behavior with coordinated speech and gesture."""
        current_time = time.time()
        if current_time - self.last_greeting_time < self.greeting_cooldown:
            return

        try:
            # Start waving gesture in a separate thread
            wave_task = self.executor.submit(wave, self.pepper)
            
            # Prepare greeting text
            if name != "Unknown":
                greeting_text = f"Hello {name}! Nice to see you!"
            else:
                greeting_text = "Hello there! Nice to meet you!"
            
            # Speak greeting
            speak_task = self.executor.submit(self.speak, greeting_text)
            
            # Wait for both tasks to complete
            wave_task.result()
            speak_task.result()
            
            self.last_greeting_time = current_time
            self.face_already_greeted = True

            #Start Rasa
            self.start_rasa_interaction()
            
        except Exception as e:
            self.logger.error(f"Error in greeting behavior: {e}")
    
    def start_rasa_interaction(self):
        """Start an interaction loop with Rasa."""
        self.logger.info("Starting Rasa interaction...")
        speak("Would you like to chat via audio or text?", "interaction_start.mp3")
        chat_choice = input("Choose interaction mode ('audiochat' or 'chatbox'): ").strip().lower()
        use_chatbox = chat_choice == 'chatbox'
        use_audio = chat_choice == 'audiochat'

        if use_chatbox:
            threading.Thread(target=open_chat_html).start()

        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        while True:
            if use_chatbox:
                user_input = input("Chatbox Input (type 'done' to switch): ").strip()
                if user_input.lower() == 'done':
                    speak("Switching to audio chat.", "switch_audio.mp3")
                    use_chatbox = False
                    use_audio = True
                    continue
            elif use_audio:
                speak("Please say something...", "audio_prompt.mp3")
                with microphone as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)

                try:
                    user_input = recognizer.recognize_google(audio)
                    print(f"User: {user_input}")
                except sr.UnknownValueError:
                    speak("Sorry, I couldn't understand that. Please try again.", "unknown_value.mp3")
                    continue
                except sr.RequestError as e:
                    speak("Sorry, there was an issue with the speech recognition service.", "request_error.mp3")
                    self.logger.error(f"Speech recognition error: {e}")
                    continue

            if user_input.lower() in ['bye', 'goodbye', 'exit']:
                speak("Goodbye!", "goodbye.mp3")
                head_nod(self.pepper)
                break

        rasa_response = get_rasa_response(user_input)
        if rasa_response:
            pepper_response = rasa_response[0].get("text", "I'm not sure I understood that.")
            print(f"Pepper: {pepper_response}")
            speak_async(pepper_response, "pepper_response.mp3")
            head_nod(self.pepper)
            

    
            
    def visualize_debug_info(self, frame, detected_faces, fps=0):
        """
        Adds debug visualization to the camera frame to help understand what's happening.
        """
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw a status box at the top
        cv2.rectangle(debug_frame, (0, 0), (width, 60), (0, 0, 0), -1)
        
        # Add system status
        status_text = "System Active - Press 'q' to quit"
        cv2.putText(debug_frame, status_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add FPS counter
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(debug_frame, fps_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detected faces with name labels
        for face in detected_faces:
            top, right, bottom, left = face['location']
            name = face['name']
            
            # Green for known faces, Red for unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw face rectangle
            cv2.rectangle(debug_frame, (left, top), (right, bottom), color, 2)
            
            # Add name label with background
            label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(debug_frame, (left, top - 20), 
                        (left + label_size[0], top), color, -1)
            cv2.putText(debug_frame, name, (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return debug_frame

    def run(self) -> None:
        """Main loop with improved visualization and performance tracking."""
        try:
            self.logger.info("Starting Pepper Interactive System...")
            
            # Variables for FPS calculation
            fps = 0
            frame_count = 0
            start_time = time.time()
            
            while True:
                loop_start = time.time()
                
                # Process camera feed
                img, detected_faces = self.process_camera_feed()
                if img is None:
                    self.logger.warning("No frame received")
                    time.sleep(0.1)  # Prevent busy waiting
                    continue
                
                
                # Update face tracker with detections
                if detected_faces:
                    is_stable, stable_name, confidence = self.face_tracker.update(detected_faces[0])
                else:
                    is_stable, stable_name, confidence = self.face_tracker.update(None)
                
                # Handle stable face detection events
                if is_stable and not self.face_already_greeted:
                    self.logger.info(f"Stable face detected: {stable_name} ({confidence:.2f})")
                    self.trigger_greeting(stable_name)
                elif not is_stable:
                    self.face_already_greeted = False
                
                # Create debug visualization
                debug_frame = self.visualize_debug_info(img, detected_faces, fps)
                
                # Show the frame
                cv2.imshow('Pepper Vision', debug_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit command received")
                    break
                elif key == ord('r'):  # Reset greeting state
                    self.logger.info("Resetting greeting state")
                    self.face_already_greeted = False
                
                # Control loop timing
                process_time = time.time() - loop_start
                if process_time < 1/30:  # Target 30 FPS
                    time.sleep(1/30 - process_time)
                    
                # Create debug visualization
                debug_frame = self.visualize_debug_info(img, detected_faces, fps)
                
                # Show the frame
                cv2.imshow('Pepper Vision', debug_frame)
                
                # Handle key presses and timing
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Update FPS calculation
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = 30 / (end_time - start_time)
                    start_time = time.time()
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down gracefully...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Release webcam
            if hasattr(self, 'cap'):
                self.cap.release()
            
            # Cleanup simulation
            self.simulation_manager.stopSimulation(self.client)
            
            # Close windows
            cv2.destroyAllWindows()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Create known_faces directory if it doesn't exist
    faces_dir = "known_faces"
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    
    # Initialize and run the system
    pepper_system = PepperInteractiveSystem(known_faces_dir=faces_dir)
    pepper_system.run()
