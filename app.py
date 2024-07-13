from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')


# Function to load and preprocess image
def preprocess_frame(frame):
    img = cv2.resize(frame, (299, 299))
    img_array = np.expand_dims(img, axis=0)
    return preprocess_input(img_array)


# Function to detect objects in frames
def detect_objects(video_path, search_query):
    capture = cv2.VideoCapture(video_path)
    frame_number = 0
    found_frames = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        # Preprocess the frame and make predictions
        img = preprocess_frame(frame)
        predictions = model.predict(img)
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        # Check if the search query is in the predictions
        for _, label, _ in decoded_predictions:
            if search_query.lower() in label.lower():
                found_frames.append((frame_number, frame))
                break

        frame_number += 1

    capture.release()
    return found_frames


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Homepage route
@app.route('/')
def index():
    return render_template('index.html')


# Route for uploading video
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Validate the file extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get user search query from form
            search_query = request.form.get('search_query')

            # Detect objects in the video
            found_frames = detect_objects(filepath, search_query)

            if found_frames:
                # Save found frames to display later
                for idx, (frame_number, frame) in enumerate(found_frames):
                    frame_filename = f'frame_{frame_number}.jpg'
                    frame_filepath = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                    cv2.imwrite(frame_filepath, frame)

                # Redirect to display frames route
                return redirect(url_for('display_frames', video_path=filename))
            else:
                flash(f'Object "{search_query}" not found in the video.')
                return redirect(request.url)

    return render_template('upload.html')


# Route to display frames with detected object
@app.route('/display_frames/<video_path>')
def display_frames(video_path):
    folder_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    frame_files = [filename for filename in os.listdir(folder_path) if filename.startswith('frame_')]
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort frames numerically
    return render_template('results.html', video_path=video_path, frame_files=frame_files)


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)
