import sqlite3
import cv2
import os
from flask import Flask,request,render_template,redirect,session,url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
# import db

#VARIABLES
MESSAGE = "WELCOME  " \
          " Instruction: to register your attendence kindly click on 'a' on keyboard"

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')

#### get a number of total registered users

def totalreg():
    return len(os.listdir('static/faces'))

def get_registered_users():
    users = []
    faces_dir = 'static/faces'
    if os.path.exists(faces_dir):
        for folder in os.listdir(faces_dir):
            if '_' in folder:
                name, user_id = folder.rsplit('_', 1)
                users.append({'name': name, 'id': user_id})
    return users

def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    from datetime import date
    datetoday = date.today().strftime("%m_%d_%y")
    filename = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(filename):
        return [], [], [], 0
    df = pd.read_csv(filename)
    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    times = df['Time'].tolist()
    l = len(df)
    return names, rolls, times, l

#### Add Attendance of a specific user
def add_attendance(name, roll):
    print(f"add_attendance called with: {name}, {roll}")  # Add this line
    from datetime import datetime
    datetoday = datetime.now().strftime("%m_%d_%y")
    filename = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('Name,Roll,Time\n')
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Writing to file:", filename)
    with open(filename, 'a') as f:
        f.write(f'\n{name},{roll},{current_time}')
    print("Write complete")

################## ROUTING FUNCTIONS ##############################

#### Our main page
@app.route('/')
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    registered_users = get_registered_users()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           registered_users=registered_users, totalreg=totalreg(),
                           datetoday2=datetoday2, mess=MESSAGE)

#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    ATTENDENCE_MARKED = False
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                cv2.putText(frame, "Press 'a' to mark attendance", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                key = cv2.waitKey(1)
                if key == ord('a'):
                    try:
                        identified_person = identify_face(face.reshape(1, -1))[0]
                        print("Predicted:", identified_person)
                        name, roll = identified_person.rsplit('_', 1)
                        cv2.putText(frame, f'{name}_{roll}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        add_attendance(name, roll)
                        ATTENDENCE_MARKED = True
                        cv2.putText(frame, "Attendance Marked!", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Attendance Check, press \"q\" to exit', frame)
                        cv2.waitKey(1000)
                        break
                    except Exception as e:
                        cv2.putText(frame, "Face not recognized", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Attendance Check, press \"q\" to exit', frame)
                        cv2.waitKey(1000)
        else:
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Attendance Check, press \"q\" to exit', frame)
        if ATTENDENCE_MARKED or cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendance taken successfully' if ATTENDENCE_MARKED else 'No attendance taken'
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return redirect(url_for('home'))


    # return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)

#### Our main function which runs the Flask App
app.run(debug=True,port=1000)
if __name__ == '__main__':
    pass
#### This function will run when we add a new user
