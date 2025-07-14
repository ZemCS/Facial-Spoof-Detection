import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import base64
from io import BytesIO
from PIL import Image, ImageOps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    max_http_buffer_size=100000000
)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'
SPOOF_THRESHOLD = 0.0001

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)
    return hist

def is_live_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return False, 0.0, "No face detected."
    (x, y, w, h) = faces[0]
    face_roi = image[y:y+h, x:x+w]
    if face_roi.size == 0:
        return False, 0.0, "Empty face region."
    lbp_features = extract_lbp_features(face_roi)
    if np.sum(lbp_features) == 0:
        return False, 0.0, "Empty LBP histogram."
    variance = np.var(lbp_features)
    is_live = variance > SPOOF_THRESHOLD
    return is_live, variance, "Success"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
@socketio.on('video_frame')
@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        img_data = base64.b64decode(data.split(',')[1])
        pil_image = Image.open(BytesIO(img_data))

        # ✅ Auto-rotate image based on EXIF
        pil_image = ImageOps.exif_transpose(pil_image)

        # Convert to OpenCV format
        img = np.array(pil_image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ✅ Save corrected image for debugging
        cv2.imwrite("received_frame.jpg", img)

        is_live, variance, message = is_live_face(img)

        if is_live:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            emit('live_face_detected', {
                'image': f'data:image/jpeg;base64,{img_base64}',
                'variance': variance,
                'message': 'Live face detected'
            })
        else:
            emit('frame_result', {'message': message, 'variance': variance})
    except Exception as e:
        emit('frame_result', {'message': f'Error: {str(e)}', 'variance': 0.0})
        
if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)    