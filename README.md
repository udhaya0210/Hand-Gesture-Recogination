📌 About the Project
This project uses Computer Vision and Machine Learning to recognize hand gestures in real-time through a webcam and converts them into spoken voice output using Text-to-Speech. It is designed to assist individuals who communicate through sign language by automatically translating gestures into audible speech.

🎯 Features
🖐️ Real-time hand gesture detection via webcam
🤖 ML model prediction for 51 unique gesture labels
🔊 Text-to-Speech conversion using pyttsx3
🦴 Hand landmark visualization using MediaPipe
⌨️ Keyboard controls — Press S to speak, Q to quit
🧠 Supports both hands simultaneously
🗂️ Supported Gestures (51 Labels)
#	Gesture	#	Gesture	#	Gesture
0	Hello	17	Name	34	Hear
1	Goodbye	18	Bad	35	Feel
2	Stop	19	Happy	36	Understand
3	Thank You	20	Good	37	Forget
4	Love	21	Friend	38	Funny
5	I	22	Home	39	Small
6	You	23	Need	40	Strong
7	Okay	24	Hate	41	Excited
8	Water	25	Which	42	Beautiful
9	Think	26	Big	43	Silence
10	Why	27	Food	44	Wait
11	What	28	Morning	45	Help
12	How	29	Speak	46	Finish
13	Where	30	Go	47	Angry
14	When	31	Learn	48	Believe
15	Sleep	32	Read	49	Dream
16	Our Name	33	Touch	50	See
🛠️ Tech Stack
Layer	Technology
👁️ Computer Vision	OpenCV
🦴 Hand Tracking	MediaPipe
🤖 ML Model	scikit-learn (Pickle)
🔢 Data Processing	NumPy
🔊 Text-to-Speech	pyttsx3
🐍 Language	Python 3.x
📁 Project Structure
hand-gesture-recognition/
│
├── model.p                  # Trained ML model (pickle file)
├── inference_classifier.py  # Main script (real-time detection)
├── data/                    # Training data (if available)
├── requirements.txt         # Dependencies
└── README.md
⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/sanjaybx1/hand-gesture-recognition
cd hand-gesture-recognition
2. Install Dependencies
pip install opencv-python mediapipe numpy pyttsx3 scikit-learn
3. Run the Project
python inference_classifier.py
⚠️ Make sure model.p is present in the root directory before running.

🎮 How to Use
Key	Action
Show hand to webcam	Gesture gets detected & label shown on screen
Press S	Converts the detected gesture to voice
Press Q	Quit the application
🔄 How It Works
📷 Webcam Input
      ↓
🦴 MediaPipe — Extracts 21 hand landmarks (x, y coordinates)
      ↓
🔢 NumPy — Normalizes landmark data (42 features per hand)
      ↓
🤖 ML Model — Predicts gesture label from features
      ↓
📝 Label displayed on screen with bounding box
      ↓
🔊 pyttsx3 — Speaks the label on 'S' key press

---

## 🚀 Future Improvements

- [ ] Add support for full sentence formation
- [ ] Train on larger custom dataset for more accuracy
- [ ] Build a GUI interface using Tkinter or PyQt
- [ ] Deploy as a web app using Flask + WebRTC
- [ ] Add regional sign language support (ISL - Indian Sign Language)
