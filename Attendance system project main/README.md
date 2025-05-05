# Face Recognition Attendance System 

A modern dark-themed attendance tracking system that uses facial recognition to identify and record user attendance.


## Features

- **Facial Recognition**: Automated attendance tracking using OpenCV and machine learning
- **Dark Theme**: Modern, eye-friendly dark user interface with animations
- **User Management**: Add and manage users with their facial data
- **Real-time Attendance**: Record attendance with a single click
- **Admin Dashboard**: View statistics, manage users, and export attendance data
- **Responsive Design**: Works on various screen sizes

## Technologies Used

- **Backend**: Python, Flask, OpenCV, scikit-learn
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Libraries**: 
  - AOS (Animate On Scroll)
  - jQuery
  - Material Icons
  - Pandas (for data processing)

## Installation

### Prerequisites

- Python 3.7+
- Webcam/camera for facial recognition
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-recognition-attendance.git
   cd face-recognition-attendance
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:1000
   ```

## Usage

### Adding New Users

1. On the main page, find the "Add New User" section
2. Enter the user's name and ID
3. Click "Add New User"
4. Follow the webcam instructions to capture facial data
5. The user will be added to the database

### Taking Attendance

1. Click the "Take Attendance" button on the home page
2. The system will open the webcam
3. When a face is detected, press the 'a' key to mark attendance
4. If the face is recognized, the attendance will be recorded
5. Press 'q' to exit the attendance mode

### Admin Access

1. Click "Admin Login" in the top-right corner
2. Enter admin credentials (default: username: "admin", password: "admin123")
3. Access the admin dashboard to view statistics and manage system settings

## Project Structure

```
face-recognition-attendance/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── haarcascade_frontalface_default.xml  # Face detection model
├── static/
│   ├── style.css           # Main dark theme styling
│   ├── adminlogin.css      # Login page styling
│   ├── images/
│   │   └── companylogo.jpg # Logo image
│   └── faces/              # User face data (auto-generated)
├── templates/
│   ├── home.html           # Main attendance page
│   ├── admin.html          # Admin dashboard
│   ├── adminlogin.html     # Admin login page
│   ├── sign.html           # User registration page
│   └── debug.html          # System debugging page
└── Attendance/             # Attendance records (auto-generated)
```

## Troubleshooting

### User Recognition Issues

- Ensure adequate lighting when registering faces and taking attendance
- If recognition accuracy is low, try deleting the user and re-adding with more varied facial expressions
- Adjust the threshold value in `app.py` if needed

### User List Display Issues

- If users disappear from the registered users list after taking attendance, check the database paths
- Visit the `/debug_folders` route to diagnose folder structure issues
- Ensure proper permissions for the `static/faces` directory

### Dark Theme Not Loading

- Clear browser cache or use incognito mode
- Verify that all CSS files are in the correct locations
- Check browser console for any CSS loading errors

## Customization

### Changing Theme Colors

Edit the CSS variables in `static/style.css`:

```css
:root {
    --bg-primary: #121212;      /* Main background color */
    --bg-secondary: #1e1e1e;    /* Secondary background */
    --accent-color: #3498db;    /* Primary accent color */
    /* ...other color variables... */
}
```

### Modifying Recognition Settings

In `app.py`, you can adjust:

- Face detection parameters in the `extract_faces()` function
- Recognition threshold in the `identify_face()` function
- Number of training images in the `add()` function

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for facial recognition capabilities
- Bootstrap for responsive design components
- AOS library for smooth animations
- Flask for the web framework
