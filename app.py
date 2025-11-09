import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # change to a secure random value for production

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(app.root_path, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_PATH = os.path.join(app.root_path, "predictions.db")
MODEL_PATH = os.path.join(app.root_path, "model.h5")

# --- Initialize database ---
def init_db():
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                correct_prediction INTEGER,
                correct_class TEXT,
                timestamp DATETIME 
            )
        """)
        connection.commit()

init_db()

# --- Load model ---
# Ensure model.h5 is present in the same folder as app.py
model = load_model(MODEL_PATH)

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    if session.get('awaiting_feedback'):
        return render_template('index.html', error="⚠️ Please provide feedback before uploading a new image.")
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if session.get('awaiting_feedback'):
        return render_template('index.html', error="⚠️ Please provide feedback before uploading a new image.")

    imagefile = request.files.get('imagefile')
    if not imagefile or imagefile.filename == '':
        return render_template('index.html', error="⚠️ No file selected")

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
    imagefile.save(image_path)

    # Preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    # Predict
    preds = model.predict(img_array)
    classes = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
    predicted_class = classes[int(np.argmax(preds))]
    confidence = round(float(np.max(preds)) * 100, 2)

    # Insert into DB with NULL feedback fields
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO predictions (filename, predicted_class, confidence, correct_prediction, correct_class, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (imagefile.filename, predicted_class, confidence, None, None))
        connection.commit()

    # Mark that we’re awaiting feedback
    session['awaiting_feedback'] = True
    session['current_filename'] = imagefile.filename

    # redirect to feedback route with query args
    feedback_url = url_for('feedback', filename=imagefile.filename, prediction=predicted_class, confidence=confidence)
    print("Redirecting to feedback page:", feedback_url)  # Debug log
    return redirect(feedback_url, code=302)

@app.route('/feedback', methods=['GET'])
def feedback():
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    return render_template('feedback.html', filename=filename, prediction=prediction, confidence=confidence)

@app.route('/feedback', methods=['POST'])
def feedback_submit():
    filename = request.form.get('filename')
    feedback = request.form.get('feedback')
    # note: correct_class may be empty string if not provided
    correct_class = request.form.get('correct_class', '').strip() or None

    # If user said "no" but didn’t select class → show error
    if feedback == 'no' and not correct_class:
        return render_template('feedback.html',
                               filename=filename,
                               prediction=request.form.get('prediction'),
                               confidence=request.form.get('confidence'),
                               error="⚠️ Please select the correct class before submitting.")

    correct_prediction = 1 if feedback == 'yes' else 0

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("""
            UPDATE predictions
            SET correct_prediction = ?, correct_class = ?
            WHERE filename = ?
        """, (correct_prediction, correct_class, filename))
        connection.commit()

    # allow next prediction
    session['awaiting_feedback'] = False
    session.pop('current_filename', None)

    return render_template('index.html', message="✅ Thank you for your feedback! You may now upload a new image.")

@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/dashboard')
def dashboard():
    # Connect to DB
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()

        # 1️⃣ Accuracy Over Time – simulate or calculate from feedback
        cursor.execute("""
            SELECT id, correct_prediction
            FROM predictions
            WHERE correct_prediction IS NOT NULL
            ORDER BY id
        """)
        records = cursor.fetchall()
        if records:
            cumulative_correct = 0
            accuracy_over_time = []
            for i, (_, correct) in enumerate(records, start=1):
                if correct == 1:
                    cumulative_correct += 1
                accuracy = round((cumulative_correct / i) * 100, 2)
                accuracy_over_time.append(accuracy)
        else:
            accuracy_over_time = []

        # 2️⃣ Prediction Distribution (per disease class)
        cursor.execute("""
            SELECT predicted_class, COUNT(*) 
            FROM predictions
            GROUP BY predicted_class
        """)
        dist_data = cursor.fetchall()
        labels = [row[0] for row in dist_data]
        counts = [row[1] for row in dist_data]

        # 3️⃣ Feedback Summary (correct vs incorrect)
        cursor.execute("""
            SELECT correct_prediction, COUNT(*) 
            FROM predictions
            WHERE correct_prediction IS NOT NULL
            GROUP BY correct_prediction
        """)
        feedback_data = cursor.fetchall()
        feedback_labels = ["Correct" if row[0] == 1 else "Incorrect" for row in feedback_data]
        feedback_counts = [row[1] for row in feedback_data]

    return render_template(
        'dashboard.html',
        accuracy_over_time=accuracy_over_time,
        dist_labels=labels,
        dist_counts=counts,
        feedback_labels=feedback_labels,
        feedback_counts=feedback_counts
    )


if __name__ == '__main__':
    print("Upload folder path:", UPLOAD_FOLDER)
    app.run(port=3000, debug=True)
