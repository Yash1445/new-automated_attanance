import sqlite3
import cv2
import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from datetime import date, datetime
import pandas as pd
import joblib
import time
import shutil

# VARIABLES
MESSAGE = "WELCOME! Instruction: to register your attendance kindly click on 'a' on keyboard"

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
            folder_path = os.path.join(faces_dir, folder)
            # Only process directories, not files
            if os.path.isdir(folder_path) and '_' in folder:
                try:
                    name, user_id = folder.rsplit('_', 1)
                    users.append({'name': name, 'id': user_id})
                except ValueError:
                    # Skip folders that don't match the expected format
                    continue
    # Sort users alphabetically by name for consistent display
    users.sort(key=lambda x: x['name'])
    return users

def debug_users_folders():
    """Helper function to debug user folders"""
    faces_dir = 'static/faces'
    debug_info = []
    
    if not os.path.exists(faces_dir):
        debug_info.append(f"Faces directory '{faces_dir}' does not exist!")
        return debug_info
        
    for item in os.listdir(faces_dir):
        item_path = os.path.join(faces_dir, item)
        if os.path.isdir(item_path):
            # Count images in the folder
            image_count = len([f for f in os.listdir(item_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            debug_info.append(f"Folder: {item}, isdir: {os.path.isdir(item_path)}, Images: {image_count}")
        else:
            debug_info.append(f"File: {item}, isdir: False")
    
    return debug_info

@app.route('/debug_folders')
def debug_folders():
    debug_info = debug_users_folders()
    return render_template('debug.html', debug_info=debug_info)

def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

def preprocess_face(face):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Resize to standard dimensions
        gray = cv2.resize(gray, (50, 50))
        
        # Normalize pixel values
        gray = gray.astype('float32') / 255.0
        
        return gray
    except Exception as e:
        print(f"Error preprocessing face: {str(e)}")
        return None

#### Identify face using ML model
def identify_face(facearray):
    try:
        # Load model
        model_path = 'static/face_recognition_model.pkl'
        if not os.path.exists(model_path):
            print("Model file doesn't exist!")
            return "unknown", 999.0
            
        model = joblib.load(model_path)
        
        # Ensure proper normalization
        if facearray.max() > 1.0:
            facearray = facearray.astype('float32') / 255.0
        
        # Get all registered users
        registered_users = [f for f in os.listdir('static/faces') if os.path.isdir(f'static/faces/{f}')]
        if len(registered_users) == 0:
            return "unknown", 999.0
        
        # Make prediction with extremely high threshold
        # This makes the system very lenient
        threshold = 15.0  # Try a higher value
        distances, indices = model.kneighbors(facearray)
        nearest_distance = distances[0][0]
        
        print(f"Recognition distance: {nearest_distance}")
        
        # If distance is too high, mark as unknown
        if nearest_distance > threshold:
            return "unknown", nearest_distance
            
        # Get prediction
        pred = model.predict(facearray)
        return pred[0], nearest_distance
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")
        return "unknown", 999.0

#### A function which trains the model on all the faces available in faces folder
def train_model():
    try:
        faces = []
        labels = []
        userlist = os.listdir('static/faces')
        
        # Ensure we have users to train on
        if len(userlist) == 0:
            print("No users to train on!")
            return False
            
        for user in userlist:
            user_folder = f'static/faces/{user}'
            if not os.path.isdir(user_folder):
                continue
                
            image_files = os.listdir(user_folder)
            if len(image_files) == 0:
                print(f"No images for user {user}")
                continue
                
            print(f"Training on {len(image_files)} images for {user}")
            
            for imgname in image_files:
                if not imgname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = f'{user_folder}/{imgname}'
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue
                    
                # Apply same preprocessing as during recognition
                processed_face = preprocess_face(img)
                if processed_face is None:
                    continue
                    
                # Flatten the face image into a 1D array
                faces.append(processed_face.ravel())
                labels.append(user)
        
        if len(faces) == 0:
            print("No faces found for training!")
            return False
            
        # Convert to numpy arrays
        faces = np.array(faces)
        
        # Create and train model
        # Use 1-NN for exact matching
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        knn.fit(faces, labels)
        
        # Save the model
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        
        print(f"Model trained successfully with {len(faces)} images from {len(set(labels))} users")
        return True
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    try:
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
    except Exception as e:
        print(f"Error extracting attendance: {str(e)}")
        return [], [], [], 0

#### Add Attendance of a specific user
def add_attendance(name, roll):
    try:
        from datetime import datetime
        datetoday = datetime.now().strftime("%m_%d_%y")
        filename = f'Attendance/Attendance-{datetoday}.csv'
        
        # Create file if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('Name,Roll,Time')
                
        # Check if already marked attendance
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if df[(df['Name'] == name) & (df['Roll'] == roll)].shape[0] > 0:
                print(f"Attendance already marked for {name}_{roll}")
                return False
                
        # Record attendance
        current_time = datetime.now().strftime("%H:%M:%S")
        with open(filename, 'a') as f:
            f.write(f'\n{name},{roll},{current_time}')
            
        print(f"Attendance marked for {name}_{roll}")
        return True
    except Exception as e:
        print(f"Error adding attendance: {str(e)}")
        return False

################## ROUTING FUNCTIONS ##############################

#### Our main page
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
    ATTENDANCE_MARKED = False
    message = ""
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                message = "Could not access camera"
                print(message)
                return render_template('home.html', names=[], rolls=[], times=[], l=0, 
                                      totalreg=totalreg(), datetoday2=datetoday2, mess=message)
        
        # Check if we have registered users
        if totalreg() == 0:
            message = "No registered users! Please register users first."
            print(message)
            cap.release()
            return render_template('home.html', names=[], rolls=[], times=[], l=0, 
                                  totalreg=totalreg(), datetoday2=datetoday2, mess=message)
        
        # Check if model exists
        if not os.path.exists('static/face_recognition_model.pkl'):
            message = "Face recognition model not found! Please register users first."
            print(message)
            cap.release()
            return render_template('home.html', names=[], rolls=[], times=[], l=0, 
                                  totalreg=totalreg(), datetoday2=datetoday2, mess=message)
        
        # Main attendance loop
        while True:
            ret, frame = cap.read()
            if not ret:
                message = "Failed to capture frame from camera"
                print(message)
                break
                
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
            # If faces detected
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Show instructions
                    cv2.putText(frame, "Press 'a' to mark attendance", (30, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Process key press
                    key = cv2.waitKey(1)
                    if key == ord('a'):
                        try:
                            # Get face region with some margin
                            y_margin = int(h * 0.1)
                            x_margin = int(w * 0.1)
                            y1 = max(0, y - y_margin)
                            y2 = min(frame.shape[0], y + h + y_margin)
                            x1 = max(0, x - x_margin)
                            x2 = min(frame.shape[1], x + w + x_margin)
                            face = frame[y1:y2, x1:x2]
                            
                            # Ensure face exists and is valid
                            if face.size == 0 or face is None:
                                print("Invalid face region")
                                continue
                                
                            # Preprocess face
                            processed_face = preprocess_face(face)
                            if processed_face is None:
                                print("Face preprocessing failed")
                                continue
                                
                            # Identify face
                            (identified_person, distance) = identify_face(processed_face.reshape(1, -1))
                            
                            print(f"Recognition result: {identified_person}, distance: {distance}")
                            
                            # Check if face is unknown
                            if identified_person == "unknown":
                                cv2.putText(frame, f"Unknown Face (d={distance:.2f})", (30, 70), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                cv2.imshow('Attendance Check, press "q" to exit', frame)
                                cv2.waitKey(2000)
                                continue
                                
                            # Process the identified person
                            if '_' in identified_person:
                                name, roll = identified_person.rsplit('_', 1)
                                user_text = f"{name} (ID: {roll})"
                                
                                # Display identified name
                                cv2.putText(frame, user_text, (x, y - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                                # Mark attendance
                                add_attendance(name, roll)
                                ATTENDANCE_MARKED = True
                                message = f"Attendance marked for {name}"
                                
                                # Show success message
                                cv2.putText(frame, "Attendance Marked!", (30, 70), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.imshow('Attendance Check, press "q" to exit', frame)
                                cv2.waitKey(2000)
                                break
                        except Exception as e:
                            message = f"Error during recognition: {str(e)}"
                            print(message)
                            cv2.putText(frame, "Recognition error", (30, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.imshow('Attendance Check, press "q" to exit', frame)
                            cv2.waitKey(2000)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
            # Show frame
            cv2.imshow('Attendance Check, press "q" to exit', frame)
            
            # Check for exit condition
            if ATTENDANCE_MARKED or cv2.waitKey(1) == ord('q'):
                break
    
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Update attendance records
        names, rolls, times, l = extract_attendance()
        registered_users = get_registered_users()  # <-- Add this line
        
        # Set final message
        if not message:
            message = 'Attendance taken successfully' if ATTENDANCE_MARKED else 'No attendance taken'
        
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, 
                              registered_users=registered_users, totalreg=totalreg(), 
                              datetoday2=datetoday2, mess=message)
    except Exception as e:
        message = f"Error: {str(e)}"
        print(message)
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        registered_users = get_registered_users()  # <-- Add this line
        return render_template('home.html', names=[], rolls=[], times=[], l=0, 
                              registered_users=registered_users, totalreg=totalreg(), 
                              datetoday2=datetoday2, mess=message)

@app.route('/add', methods=['GET', 'POST'])
def add():
    try:
        # Get form data
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        
        # Validate input
        if not newusername or not newuserid:
            print("Username or ID is empty")
            return redirect(url_for('home'))
            
        # Create folder path
        userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
        
        # Remove existing folder if it exists
        if os.path.isdir(userimagefolder):
            shutil.rmtree(userimagefolder)
            print(f"Removed existing folder: {userimagefolder}")
            
        # Create new folder
        os.makedirs(userimagefolder)
        print(f"Created folder: {userimagefolder}")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("Could not access camera")
                return redirect(url_for('home'))
        
        # Capture images
        i, j = 0, 0
        while i < 25:  # Capture more images for better training
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Detect faces
            faces = extract_faces(frame)
            
            for (x, y, w, h) in faces:
                # Create larger face region
                y_margin = int(h * 0.2)
                x_margin = int(w * 0.2)
                y1 = max(0, y - y_margin)
                y2 = min(frame.shape[0], y + h + y_margin)
                x1 = max(0, x - x_margin)
                x2 = min(frame.shape[1], x + w + x_margin)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 20), 2)
                cv2.putText(frame, f'Images: {i}/25', (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 20), 2)
                
                # Capture image at interval (every 5 frames)
                if j % 5 == 0 and i < 25:
                    # Extract face region
                    face = frame[y1:y2, x1:x2]
                    
                    # Save original image
                    img_name = f"{newusername}_{i}.jpg"
                    img_path = os.path.join(userimagefolder, img_name)
                    cv2.imwrite(img_path, face)
                    
                    # Save with slight variations for better training
                    if i % 3 == 0:
                        # Save slightly brighter version
                        bright = cv2.convertScaleAbs(face, alpha=1.1, beta=10)
                        cv2.imwrite(os.path.join(userimagefolder, f"{newusername}_bright_{i}.jpg"), bright)
                        
                        # Save slightly darker version
                        dark = cv2.convertScaleAbs(face, alpha=0.9, beta=-10)
                        cv2.imwrite(os.path.join(userimagefolder, f"{newusername}_dark_{i}.jpg"), dark)
                    
                    print(f"Saved image {i}: {img_name}")
                    i += 1
                    
                    # Show success message
                    cv2.putText(frame, "Image captured!", (30, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('Adding new User', frame)
                    cv2.waitKey(200)  # Slight pause
                
                j += 1
            
            # Show frame
            cv2.imshow('Adding new User', frame)
            
            # Check for exit
            if cv2.waitKey(1) == 27 or i >= 25:  # ESC key
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Train model with new data
        print("Training model with new user data...")
        train_model()
        
        return redirect(url_for('home'))
    except Exception as e:
        print(f"Error in add user: {str(e)}")
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        return redirect(url_for('home'))

@app.route('/admin')
def admin():
    # Get attendance data
    names, rolls, times, l = extract_attendance()
    registered_users = get_registered_users()
    
    # Count daily attendance, absences, and late arrivals
    daily_attendance = l
    absent_count = totalreg() - l
    
    # Count late arrivals (assuming 9:30 AM is the cutoff)
    late_count = 0
    for time in times:
        if time > "09:30:00":
            late_count += 1
    
    return render_template('admin.html', 
                          names=names, 
                          rolls=rolls, 
                          times=times, 
                          l=l,
                          registered_users=registered_users, 
                          totalreg=totalreg(),
                          daily_attendance=daily_attendance,
                          absent_count=absent_count,
                          late_count=late_count,
                          datetoday2=datetoday2)

@app.route('/adminlogin', methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        username = request.form.get('userName')
        password = request.form.get('password')
        
        # Here you would add proper authentication logic
        # For a simple example:
        if username == "admin" and password == "admin123":
            return redirect(url_for('admin'))
        else:
            return render_template('adminlogin.html', error="Invalid credentials")
    
    return render_template('adminlogin.html')
# ...existing code...

@app.route('/admin/export')
def export_data():
    # Export attendance data logic here
    return redirect(url_for('admin'))

@app.route('/admin/retrain')
def retrain_model():
    # Call your train_model function
    train_model()
    return redirect(url_for('admin'))

# ...existing code...

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True, port=1000)