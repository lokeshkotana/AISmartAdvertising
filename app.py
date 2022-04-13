from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import time
from flask import Flask, request, jsonify, render_template, Response


app = Flask(__name__)
model = load_model('gender_detection.model')
webcam = cv2.VideoCapture("people2.mp4")    
# webcam = cv2.VideoCapture(0)
classes = ['man','woman']



@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict')
def predict():

    # # open webcam
    # webcam = cv2.VideoCapture("people2.mp4")
    
    # classes = ['man','woman']

    # loop through frames
    while webcam.isOpened():

        # read frame from webcam 
        status, frame = webcam.read()

        # apply face detection
        face, confidence = cv.detect_face(frame)
        
        people = ""
  

        # loop through detected faces
        for idx, f in enumerate(face):

            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

            # crop the detected face region
            face_crop = np.copy(frame[startY:endY,startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]

            # label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
            people = people + ' ' + label

        if len(people)>0:
            print(people)
        

        # display output
        # cv2.imshow("gender detection", frame)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.1)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    # release resources
    # webcam.release()
    # cv2.destroyAllWindows()
    
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(predict(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test', methods = ['GET'])
def test():    
    # open webcam
        # loop through frames
    if webcam.isOpened():

        # read frame from webcam 
        status, frame = webcam.read()

        # apply face detection
        face, confidence = cv.detect_face(frame)
        
        people = []
  

        # loop through detected faces
        for idx, f in enumerate(face):

            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

            # crop the detected face region
            face_crop = np.copy(frame[startY:endY,startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]

            # label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
            people.append(label)

        if len(people)>0:
            return jsonify(result = people )
           
    


if __name__ == "__main__":
    app.run(debug=True)
