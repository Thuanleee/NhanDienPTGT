import os
import cv2
import torch
from flask import Flask, render_template, request, send_from_directory, Response
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd
import random
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Cấu hình Flask
app = Flask(__name__)

# Cấu hình thư mục tải lên
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Khóa bảo mật Flask
app.secret_key = 'your_secret_key'

# Cấu hình Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Tải mô hình YOLOv8 lên GPU nếu có
MODEL_PATH = 'data/bester.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH).to(device)
print(f"Thiết bị đang sử dụng: {device}")

# Danh sách người dùng
users = {
    "admin": generate_password_hash("admin"),
    "user": generate_password_hash("123"),
}

# Lớp người dùng
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Dự đoán hình ảnh với YOLOv8
def predict_image(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img)
    boxes = results[0].boxes

    pandas_df = pd.DataFrame()
    if boxes is not None:
        pandas_df = pd.DataFrame(boxes.xywh.cpu().numpy(), columns=["x_center", "y_center", "width", "height"])
        pandas_df['confidence'] = boxes.conf.cpu().numpy()
        pandas_df['class'] = boxes.cls.cpu().numpy()

    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    img_with_boxes = results[0].plot()
    cv2.imwrite(output_image_path, img_with_boxes)

    return pandas_df, os.path.basename(output_image_path), len(boxes)

# Dự đoán video với YOLOv8
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = 'result_' + os.path.basename(video_path)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_objects = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_with_boxes = results[0].plot()
        total_objects += len(results[0].boxes)
        out.write(frame_with_boxes)

    cap.release()
    out.release()

    return {
        'video_path': output_filename,
        'total_objects': total_objects,
        'total_frames': total_frames
    }

# Xử lý camera trực tiếp
def predict_camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_with_boxes = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Routes Flask
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            predictions, result_image, ndet = predict_image(img_path)

            return render_template('index.html',
                                   user_image=f'uploads/{result_image}',
                                   msg='Nhận diện thành công!',
                                   ndet=ndet,
                                   rand=random.random())

    return render_template('index.html', user=current_user.id)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/camera')
@login_required
def camera():
    return Response(predict_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video', methods=['POST'])
@login_required
def video():
    file = request.files['file']
    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        result = predict_video(video_path)

        return render_template('result_video.html',
                               video_filename=result['video_path'],
                               object_count=result['total_objects'],
                               total_frames=result['total_frames'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            return render_template('index.html', msg="Đăng nhập thành công!", user=current_user.id)
        else:
            return render_template('login.html', msg="Tài khoản hoặc mật khẩu không đúng.")

    return render_template('login.html', msg="")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return render_template('index.html', msg="Đã đăng xuất thành công!")

if __name__ == '__main__':
    app.run(debug=True)
