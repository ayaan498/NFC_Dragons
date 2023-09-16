from flask import Flask,render_template,Response,session
from PIL import ImageGrab
import pyautogui
import cv2
import pickle
import mediapipe as mp
import numpy as np
from flask_socketio import SocketIO
import time
import pyscreenshot as ImageGrab
import schedule
from datetime import datetime

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
    # no_of_labels = 10
    # d = {}
    # for i in range(no_of_labels):
    #     d[i] = (0,0)
    # count = 0

    while True:
        # count += 1
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
                    # prediction probability
                    # prediction_probability = model.predict_proba([np.asarray(data_aux)])
                    # check = int(prediction[0])  
                    
                    predicted_sign = labels_dict[int(prediction[0])]
                    
                    if lastElement != predicted_sign:
                        lastElement = predicted_sign
                        closedCaptions(predicted_sign)
                    data_aux = []
                    x_ = []
                    y_ = []

                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q'):
                #     break

            # res = []
            # for i in range(no_of_labels):
            #     if d[i][1] == 0:
            #         res.append(0)
            #     else:
            #         res.append(d[i][0]/count)
            # # Filter out the non-zero accuracy values and format them into a string prompt
            # problem_context = "This is a sign language recognition system with the goal of assisting individuals with hearing impairments."
            # non_zero_accuracies = [
            #     f"For the label '{labels_dict[i]}', the accuracy is {accuracy * 1000:.2f}%."
            #     for i, accuracy in enumerate(res) if accuracy > 0
            # ]
            # prompt =problem_context+" "+ " ".join(non_zero_accuracies)
            # print(prompt)
            # predictionProbability(prompt)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def take_screenshot():
    image_name = f"screenshot-{str(datetime.now())}"
    image_name = image_name.replace(".","-")
    image_name = image_name.replace(":","-")
    image_name = image_name.replace(",","-")
    screenshot = ImageGrab.grab(bbox=(500,300,1500,800))
    filepath = f"./screenshots/{image_name}.jpg"
    screenshot.save(filepath)
    return filepath

def generate_frames_video_call():
    lastElement = "string"
        
    while True:
        # Create a video capture object for the screen
        time.sleep(1)
        screen_capture = take_screenshot()

        data_aux = []
        x_ = []
        y_ = []
        
        frame = cv2.cvtColor(cv2.imread(screen_capture), cv2.COLOR_RGB2BGR)
        H, W, _ = frame.shape
        results = hands.process(frame)

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
                
                predicted_sign = labels_dict[int(prediction[0])]
                
                if lastElement != predicted_sign:
                    lastElement = predicted_sign
                    print(predicted_sign)
                    closedCaptions(predicted_sign)
                data_aux = []
                x_ = []
                y_ = []

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/')
def index():
    return render_template('meet.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videoCall')
def videoCall():
    return Response(generate_frames_video_call(),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('closedCaptions')
def closedCaptions(caption):
    socketio.emit('closedCaptions', caption)

@socketio.on('predictionProbability')
def predictionProbability(prompt):
    socketio.emit('predictionProbability', prompt)

if __name__=="__main__":
    socketio.run(app, debug=True)

