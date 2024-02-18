from flask import Flask
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('./model.keras')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
# actions = ['Blind', 'Deaf', 'beautiful', 'cold', 'cool', 'fast', 'happy', 'hot', 'large', 'loud', 'narrow', 'new', 'quiet', 'sad', 'slow', 'small', 'warm']
# REAL TIME DETECTION / EXISTING VID

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
actions = ['again', 'alive', 'ask', 'bad', 'boy', 'busy', 'can', 'come', 'deaf', 'different', 
           'drink', 'drive', 'family', 'feel', 'few', 'find', 'finish', 'food', 'forget', 
           'friend', 'girl', 'good', 'goodnight', 'happy birthday to you', 'hearing', 'hello',
           'help', 'home', 'how', 'how are you', "it's okay", 'know', 'later', 'like', 
           'little', 'many', 'may', 'meet', 'mistake', 'more', 'my', 'name', 'new', 'old', 
           'please', 'remember', 'same', 'school', 'see', 'thank you', 'understand', 'what', 
           'when', 'where', 'which', 'who', 'why', 'work', 'yes', 'you']
sequence = []
sentence = []
predictions = []
threshold = 0.5

sequence = []
sentence = []
predictions = []
threshold = 0.5

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()

    # Display the blue prediction bar at the top
    cv2.rectangle(output_frame, (0, 0), (640, 40), (245, 117, 16), -1)

    # Display the predicted action text
    cv2.putText(output_frame, actions[np.argmax(res)], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

mp_drawing = mp.solutions.drawing_utils

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )
    

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
num_keypoints = 258  # Update the number of keypoints
# Number of frames to wait before displaying the prediction
delay_frames = 5  # Adjust this value as needed

def fun():

    cap = cv2.VideoCapture(0) # 0 for webcam, 'path' for video file
    frame_count = 0

    prid_main = []

    prid_temp = []

    while cap.isOpened():
        # Reset sentence and predictions at the beginning of each loop iteration
        sentence = []
        predictions = []

        # Read feed
        ret, frame = cap.read()
        # frame = cv2.flip(frame,1)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Prediction logic
        sequence = []
        keypoints = extract_keypoints(results)
        sequence = np.append(sequence, keypoints)
        sequence = sequence[-5 * num_keypoints:]

        if len(sequence) == 5 * num_keypoints:
            sequence = sequence.reshape((5, num_keypoints))  # Reshape the sequence
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            res = model.predict(sequence)[0]

            predictions.append(np.argmax(res))

            # Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    frame_count += 1

                    if frame_count >= delay_frames:
                        print(actions[np.argmax(res)])
                        sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        # Viz probabilities
                        image = prob_viz(res, actions, image, colors)
                else:
                    frame_count = 0  # Reset frame count if the prediction confidence is below threshold
            else:
                frame_count = 0  # Reset frame count if predictions are not consistent


        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def hello():
    fun()
    return 'hello'
