from pickle import FALSE, TRUE
from flask import Flask,render_template,Response,request
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import csv
import copy
import mediapipe as mp
import numpy as np
from app_files.main.draw import draw_info_text_word
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_info_text_word, draw_landmarks, get_args, pre_process_landmark
from logging_csv import logging_csv,getLastRowValue

app=Flask(__name__)

global createVariable
createVariable=""


global numberCSV
numberCSV=30

global Str
Str = ""

#global GlobalStr
GlobalStr = ""

global cvVariable
cvVariable=""

def generate_frames():
   
    
    use_static_image_mode = os.getenv('USE_STATIC_IMAGE_MODE', 'False').lower() in ['true', '1', 't', 'y', 'yes']

  
    device = os.getenv('DEVICE', '/dev/video0')
    camera=cv2.VideoCapture(device)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()
    cap_width = int(os.getenv('WIDTH', 640))
    cap_height = int(os.getenv('HEIGHT', 480))
    min_detection_confidence = float(os.getenv('MIN_DETECTION_CONFIDENCE', 0.5))
    min_tracking_confidence = float(os.getenv('MIN_TRACKING_CONFIDENCE', 0.5))

    cap = camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1) 
        debug_image = copy.deepcopy(image)
        # print(debug_image.shape)
        # cv.imshow("debug_image",debug_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                debug_image = draw_landmarks(debug_image, landmark_list)
                #print(keypoint_classifier_labels[hand_sign_id])
                global cvVariable
                cvVariable=keypoint_classifier_labels[hand_sign_id]
                debug_image = draw_info_text(
                    debug_image,
                    handedness,
                    cvVariable)
                debug_image = draw_info_text_word(
                    debug_image,
                    handedness,
                    getGlobalVariable())
        ret,buffer=cv2.imencode('.jpg',debug_image)
        debug_image=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + debug_image + b'\r\n')
    del(camera)
        #cv2.putText(frame, sign)
    
def draw_styled_landmarks(image, results):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
    # # Draw pose connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
    #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    #                          ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )                   
#def draw_landmarks(image, results):
  #  mp_holistic = mp.solutions.holistic # Holistic model
  #  mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
  #  mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    #lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return rh

def generate_frames_for_create():

    camera=cv2.VideoCapture(0)
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    mode = 1
    number =29
    
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        
        if 48 == key:
            number = 0
            
        ret, image = cap.read()

        if not ret:
            break
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        #cv.imshow("results",image)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks :                               
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                #print(pre_processed_landmark_list)
                #print(mode)
                #print(number)
                logging_csv(numberCSV, mode, pre_processed_landmark_list)
                # logging_csv(number, mode, landmark_list)
                debug_image = draw_landmarks(debug_image, landmark_list)
                info_text="Press key 0 to capture"
                cv2.putText(debug_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (196, 161, 33), 1, cv2.LINE_AA)
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        ret,buffer=cv2.imencode('.jpg',debug_image)
        debug_image=buffer.tobytes()
                #cv2.putText(frame, sign)
        yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + debug_image + b'\r\n')
    del(camera)                            
    #camera.release()
    #cv2.destroyAllWindows()
@app.route('/addGlobalVariable')   
def addGlobalVariable():
    global GlobalStr
    if(cvVariable!=' '):
        GlobalStr+=cvVariable
    return TRUE

@app.route('/removeGlobalVariable')   
def removeGlobalVariable():
    global GlobalStr
    GlobalStr = GlobalStr[:-1]
    #print("calleddd")
    #print(GlobalStr)
    return TRUE

@app.route('/addSpaceGlobalVariable')   
def addSpaceGlobalVariable():
    global GlobalStr
    GlobalStr+=" "
    #print("calleddd")
    #print(GlobalStr)
    return TRUE


@app.route('/addGlobalVariable')   
def getGlobalVariable():
    global GlobalStr 
    return GlobalStr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vid')
def vid():
    #print(GlobalStr)
    return render_template('vid.html',value=GlobalStr)

@app.route('/create', methods =["GET", "POST"])
def create():
    if request.method == "POST":
        Gesture = request.form.get('Gesture', '')
        Number = getLastRowValue()
        Number+=1
        global createVariable
        createVariable=Gesture
        global numberCSV
        numberCSV=Number
        csv_path = 'model/keypoint_classifier/keypointTempAppOrg.csv'
        with open(csv_path, 'a', newline="") as f:

            #print(landmark_list)
            writer = csv.writer(f)
            writer.writerow([createVariable])
        return render_template('create.html')

    return render_template('createVariable.html')

@app.route('/exp')
def exp():
    global Str
    #Str="Lorem ipsum dolor sit amet, consectetur adipisicing elit. Mollitia neque assumenda ipsam nihil, molestias magnam, recusandae quos quis inventore quisquam velit asperiores, vitae? Reprehenderit soluta, eos quod consequuntur itaque. Nam."
    return render_template('exp.html',value=GlobalStr)

@app.route('/export')
def Export():
    global GlobalStr
    f = open("SLR.txt", 'w')
    f.write(GlobalStr)
    f.close()
    print("File saved as SLR.txt!!")
    return "True"

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/createGesture')
def createGesture():
    return Response(generate_frames_for_create(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

