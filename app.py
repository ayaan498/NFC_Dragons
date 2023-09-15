from flask import Flask,render_template,Response,session
import cv2
import pickle
import mediapipe as mp
import numpy as np
from flask_socketio import SocketIO

app=Flask(__name__)
socketio = SocketIO(app)

app.secret_key = 'your_secret_key'
camera=cv2.VideoCapture(0)
model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)
labels_dict = {0: "hello", 1: "i love you", 2: "yes", 3: "good", 4: "bad", 5: "okay", 6: "you", 7: "i/i'm", 8: "why", 9: "no"}


def generate_frames():
    lastElement = "string"
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            data_aux = []
            x_ = []
            y_ = []
            # ret, frame = capture.read()
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

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10
                    prediction = model.predict([np.asarray(data_aux)])
                    # print(prediction)
                    
                    predicted_sign = labels_dict[int(prediction[0])]
                    # print(f"Predicted Sign: {predicted_sign}")
                    
                    if lastElement != predicted_sign:
                        lastElement = predicted_sign
                        print(predicted_sign)
                        closedCaptions(predicted_sign)

                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    # cv2.putText(frame, predicted_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    data_aux = []
                    x_ = []
                    y_ = []

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('translator.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('closedCaptions')
def closedCaptions(caption):
    socketio.emit('closedCaptions', caption)

if __name__=="__main__":
    socketio.run(app, debug=True)

