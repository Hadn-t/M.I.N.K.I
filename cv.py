import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import subprocess
import time
import threading
import queue

# Initialize mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class CameraStream:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_flag = False
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_frames(self):
        command = [
            "libcamera-vid",
            "--output", "-",
            "--width", "640",
            "--height", "480",
            "--timeout", "0",
            "--framerate", "30",
            "--codec", "mjpeg",
            "--brightness", "0.3",
            "--contrast", "1.5",
            "--saturation", "0.9",
            "--sharpness", "1.2",
            "--gain", "2",
            "--awb", "auto",
            "--exposure", "normal"
        ]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        while not self.stop_flag:
            try:
                size_data = process.stdout.read(4)
                if not size_data:
                    break
                    
                size = int.from_bytes(size_data, byteorder='little', signed=False)
                jpeg_data = process.stdout.read(size)
                
                if not jpeg_data:
                    break
                    
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame)
                        except queue.Empty:
                            pass
            except Exception as e:
                print(f"Error in camera capture: {e}")
                time.sleep(0.1)

    def read(self):
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self):
        self.stop_flag = True
        self.capture_thread.join()

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results is None or image is None:
        return
        
    try:
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )
            
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
            
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
            
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    except Exception as e:
        print(f"Error drawing landmarks: {e}")

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

def real_time_detection():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    
    print("Loading model...")
    model = load_model('action1.h5')
    print("Model loaded successfully")

    actions = np.array(['thanks', 'help', 'you-good', 'please'])
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 66, 226)]

    print("Initializing camera...")
    camera = CameraStream()
    time.sleep(2)  # Give camera time to warm up
    print("Camera initialized")
    
    print("Starting MediaPipe Holistic...")
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        print("MediaPipe Holistic started")
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to get frame")
                continue

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Prediction logic
            if results:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep last 30 frames

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    # Visualization logic
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Visualize probabilities
                    image = prob_viz(res, actions, image, colors)

            # Add the sentence display
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the processed frame
            cv2.imshow('Sign Language Detection', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting real-time detection...")
    real_time_detection()
