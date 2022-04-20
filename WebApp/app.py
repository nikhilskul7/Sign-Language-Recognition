from pickle import FALSE, TRUE
from flask import Flask,render_template,Response
import cv2
import os
import mediapipe as mp
import numpy as np
from keras.models import load_model

app=Flask(__name__)
camera=cv2.VideoCapture(0)

global Str
Str = ""

global createVariable
createVariable=""

def generate_frames():
    global Str
    sequence=[]
    flag=FALSE
    model=load_model('action.h5')
    actions=['A', 'B','C','D','E']
    signStr=" "
       
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    while True:
        
        mp_holistic = mp.solutions.holistic # Holistic model
        mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        ## read the camera frame
        success,frame=camera.read()
        holistic=mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
       
        #sequence=[]
       
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-20:]
        #cv2.putText(image,"actions[np.argmax(sign)]", bottomLeftCornerOfText, font, fontScale,fontColor,thickness,lineType)
        
        if len(sequence)==20:
            sign = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(sign)])
            signStr+=' '
            signStr+=str(actions[np.argmax(sign)])
            Str = signStr
            #print('here')
            sequence=[]
            flag=TRUE


        cv2.putText(image,signStr, bottomLeftCornerOfText, font, fontScale,fontColor,thickness,lineType)
            
        if not flag:
            continue
        else:
            ret,buffer=cv2.imencode('.jpg',image)
            image=buffer.tobytes()
        #cv2.putText(frame, sign)
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

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
def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

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
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    action="Temp"
    DATA_PATH=os.path.join('MP_DATA')
    holistic=mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    no_sequences=20
    flag=FALSE
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


    for frame_num in range(no_sequences):
                #print(no_sequences)
                print(frame_num)

                # Read feed
                ret, frame = camera.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, frame_num), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(5000)
                else: 
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(5000)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
               
                npy_path = os.path.join(DATA_PATH, action, str(frame_num),str(frame_num))
                print(npy_path)
                #print("saved for {}",no_sequences)
                np.save(npy_path, keypoints)
                
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                ret,buffer=cv2.imencode('.jpg',image)
                image=buffer.tobytes()
                #cv2.putText(frame, sign)
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                                
    #camera.release()
    #cv2.destroyAllWindows()
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vid')
def vid():
    return render_template('vid.html')

@app.route('/create')
def create():
    return render_template('create.html')

@app.route('/exp')
def exp():
    global Str
    #Str="Lorem ipsum dolor sit amet, consectetur adipisicing elit. Mollitia neque assumenda ipsam nihil, molestias magnam, recusandae quos quis inventore quisquam velit asperiores, vitae? Reprehenderit soluta, eos quod consequuntur itaque. Nam."
    return render_template('exp.html',value=Str)

@app.route('/export')
def Export():
    global Str
    f = open("SLR.txt", 'w')
    f.write(Str)
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

