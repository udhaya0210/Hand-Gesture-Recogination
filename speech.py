


import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

labels_dict = {0: 'hello',1:'goodbye',2:'stop',3:'Thankyou',4:'love',5:'i',6:'you',7:'okay',8:'water',9:'think',10:'why',11:'what',12:'how',13:'where',14:'when',15:'sleep',16:'our name',17:'name',18:'Bad',19:'happy',20:'good',21:'friend',22:'home',23:'need',24:'hate',25:'which',26:'big',27:'food',28:'morning',29:'speak',30:'Go',31:'Learn',32:'Read',33:'Touch',34:'Hear',35:'Feel',36:'Understand',37:'Forget',38:'Funny',39:'Small',40:'Strong',41:'Excited',42:'Beautiful',43:'Silence',44:'Wait',45:' Help',46:'Finish',47:'Angry',48:'Believe',49:'Dream',50:'See'}

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

voice_generated = True  # Initialize to True

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Couldn't read frame from the camera")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure data_aux has exactly 42 features (adjust if necessary)
            data_aux_subset = data_aux[:42]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux_subset)])

            predicted_character = labels_dict[int(prediction[0])]

            # Display the recognized character
            #print(predicted_character)

            # Convert the recognized character to speech
            if not voice_generated:
                engine.say(predicted_character)
                engine.runAndWait()
                voice_generated = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Wait for 's' key to be pressed to generate voice
    key = cv2.waitKey(1)
    if key == ord('s'):
        voice_generated = False  # Reset the flag when 's' is pressed

    # Check for 'q' key to exit the loop
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
