import cv2
import time
import threading
import winsound
import numpy as np
from datetime import datetime

class DrowsinessDetector:
    """
    A class for detecting drowsiness through eye closure detection using webcam.
    """
    
    def __init__(self):
        # Configuration parameters
        self.EYE_CLOSED_THRESHOLD = 2.0  # seconds before alarm triggers
        self.FACE_CONFIDENCE = 1.1       # face detection parameters
        self.EYE_CONFIDENCE = 1.1        # eye detection parameters
        self.ALARM_FREQUENCY = 2000      # Hz (higher for louder alarm)
        self.ALARM_DURATION = 1500       # ms
        
        # Load pre-trained models using absolute paths
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Runtime variables
        self.eye_closed_start_time = None
        self.alarm_triggered = False
        self.is_running = False
        self.alarm_thread = None
        
        # Statistics
        self.drowsiness_events = 0
        self.start_time = None
        
    def play_alarm(self):
        """Plays alarm sound in a separate thread to avoid blocking the main thread."""
        while self.alarm_triggered and self.is_running:
            winsound.Beep(self.ALARM_FREQUENCY, self.ALARM_DURATION)
            time.sleep(0.5)  # Brief pause between alarm sounds
    
    def start_alarm(self):
        """Starts the alarm in a separate thread."""
        if not self.alarm_triggered:
            self.alarm_triggered = True
            self.drowsiness_events += 1
            self.alarm_thread = threading.Thread(target=self.play_alarm)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
    
    def stop_alarm(self):
        """Stops the currently playing alarm."""
        self.alarm_triggered = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1.0)  # Wait for thread to finish
    
    def detect_eyes(self, face_roi_gray):
        """Detects eyes in the given region of interest."""
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, 
            scaleFactor=self.EYE_CONFIDENCE, 
            minNeighbors=5,
            minSize=(20, 20)
        )
        return eyes
    
    def process_frame(self, frame):
        """
        Processes a video frame for drowsiness detection.
        Returns the processed frame with annotations.
        """
        # Create a copy of the frame to avoid modifying the original
        display_frame = frame.copy()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with improved parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.FACE_CONFIDENCE,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        eyes_detected = False
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Define regions of interest for eyes (upper half of face)
            face_roi_gray = gray[y:y+int(h/2), x:x+w]
            face_roi_color = display_frame[y:y+int(h/2), x:x+w]
            
            # Detect eyes in the face region
            eyes = self.detect_eyes(face_roi_gray)
            
            # Process detected eyes
            if len(eyes) >= 1:  # At least one eye detected
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    # Draw eye rectangles
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        # Update drowsiness state
        current_time = time.time()
        
        # Handle eye state changes
        if eyes_detected:
            # Eyes are open
            self.eye_closed_start_time = None
            if self.alarm_triggered:
                self.stop_alarm()
        else:
            # No eyes detected - potentially closed
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
            elif not self.alarm_triggered and (current_time - self.eye_closed_start_time) > self.EYE_CLOSED_THRESHOLD:
                self.start_alarm()
        
        # Add status information to the frame
        self.add_status_info(display_frame, eyes_detected)
        
        return display_frame
    
    def add_status_info(self, frame, eyes_detected):
        """Adds status information to the display frame."""
        # Display eye status
        status = "Eyes Open" if eyes_detected else "Eyes Closed"
        color = (0, 255, 0) if eyes_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display alarm status if triggered
        if self.alarm_triggered:
            # Make the ALARM text more noticeable
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Add flashing effect to the frame border when alarm is active
            if int(time.time() * 2) % 2 == 0:  # Flash at 2 Hz
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
        
        # Display session statistics
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            minutes, seconds = divmod(elapsed, 60)
            cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", 
                       (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        cv2.putText(frame, f"Drowsiness Events: {self.drowsiness_events}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def run(self):
        """Main loop for capturing and processing video frames."""
        # Initialize webcam
        webcam = cv2.VideoCapture(0)
        
        # Check if webcam is opened successfully
        if not webcam.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set webcam properties for better quality
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        self.start_time = time.time()
        
        print("Drowsiness detection started. Press 'q' to quit.")
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = webcam.read()
                
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Process the frame
                display_frame = self.process_frame(frame)
                
                # Display the resulting frame
                cv2.imshow('Drowsiness Detection System', display_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error occurred: {e}")
        
        finally:
            # Clean up
            self.is_running = False
            if self.alarm_triggered:
                self.stop_alarm()
            webcam.release()
            cv2.destroyAllWindows()
            print("Drowsiness detection stopped.")
            
            # Display session summary
            session_duration = int(time.time() - self.start_time)
            minutes, seconds = divmod(session_duration, 60)
            print(f"\nSession Summary:")
            print(f"Duration: {minutes} minutes, {seconds} seconds")
            print(f"Drowsiness events detected: {self.drowsiness_events}")
            print(f"Session ended at: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()