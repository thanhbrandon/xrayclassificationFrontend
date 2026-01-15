import os
import sqlite3
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, session
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------------------------------------------------------
# Flask App Configuration
# -----------------------------------------------------------------------------
app = Flask(__name__)

# NOTE: Change this key for production — it protects sessions/cookies
app.secret_key = 'supersecretkey'

# Define important paths
UPLOAD_FOLDER = os.path.join(app.root_path, 'images')
DB_PATH = os.path.join(app.root_path, "predictions.db")
MODEL_PATH = os.path.join(app.root_path, "model.h5")

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------------------------------------------------------
# Database Setup
# -----------------------------------------------------------------------------
def init_db():
    """Creates the predictions database and table if they don’t exist."""
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()

# Initialize the DB once when app starts
init_db()

# -----------------------------------------------------------------------------
# Load the trained CNN model
# -----------------------------------------------------------------------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# -----------------------------------------------------------------------------
# Home Page (Upload + Predict)
# -----------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    """
    Display the home page.
    If user hasn’t provided feedback for last prediction,
    prevent new upload until feedback is given.
    """
    if session.get('awaiting_feedback'):
        return render_template('index.html', error="⚠️ Please provide feedback before uploading a new image.")
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    """
    Handles image upload, preprocessing, prediction, and DB logging.
    Then redirects user to feedback page.
    """
    if session.get('awaiting_feedback'):
        return render_template('index.html', error="⚠️ Please provide feedback before uploading a new image.")

    imagefile = request.files.get('imagefile')
    if not imagefile or imagefile.filename == '':
        return render_template('index.html', error="⚠️ No file selected")

    # --- Save uploaded image ---
    image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
    imagefile.save(image_path)

    # --- Preprocess image for model ---
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    # --- Predict with model ---
    preds = model.predict(img_array)
    classes = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
    predicted_class = classes[int(np.argmax(preds))]
    confidence = round(float(np.max(preds)) * 100, 2)

    # --- Insert into database ---
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                filename, predicted_class, confidence,
                correct_prediction, correct_class, timestamp
            ) VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (imagefile.filename, predicted_class, confidence, None, None))
        connection.commit()

    # --- Track that user must give feedback ---
    session['awaiting_feedback'] = True
    session['current_filename'] = imagefile.filename

    # Redirect to feedback route
    feedback_url = url_for('feedback', filename=imagefile.filename,
                           prediction=predicted_class, confidence=confidence)
    print("Redirecting to feedback page:", feedback_url)  # Debug log
    return redirect(feedback_url, code=302)

# -----------------------------------------------------------------------------
# Feedback Page
# -----------------------------------------------------------------------------
@app.route('/feedback', methods=['GET'])
def feedback():
    """Display prediction feedback page."""
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    return render_template('feedback.html',
                           filename=filename,
                           prediction=prediction,
                           confidence=confidence)


@app.route('/feedback', methods=['POST'])
def feedback_submit():
    """
    Handles user feedback.
    - If "yes": marks prediction as correct.
    - If "no": requires correct class.
    Updates DB and unlocks upload for next image.
    """
    filename = request.form.get('filename')
    feedback = request.form.get('feedback')
    correct_class = request.form.get('correct_class', '').strip() or None

    # Validation: if incorrect but no correct class given
    if feedback == 'no' and not correct_class:
        return render_template(
            'feedback.html',
            filename=filename,
            prediction=request.form.get('prediction'),
            confidence=request.form.get('confidence'),
            error="Please select the correct class before submitting."
        )

    correct_prediction = 1 if feedback == 'yes' else 0

    # Update database record
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("""
            UPDATE predictions
            SET correct_prediction = ?, correct_class = ?
            WHERE filename = ?
        """, (correct_prediction, correct_class, filename))
        connection.commit()

    # Unlock new upload
    session['awaiting_feedback'] = False
    session.pop('current_filename', None)

    return render_template(
        'index.html',
        message="Thank you for your feedback! You may now upload a new image."
    )

# -----------------------------------------------------------------------------
# Dashboard Visualization
# -----------------------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    """
    Displays dashboard with:
      1. Accuracy Over Time (line chart)
      2. Prediction Distribution (pie chart)
      3. Feedback Summary (bar chart)
    """
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()

        # --- Accuracy Over Time ---
        cursor.execute("""
            SELECT id, correct_prediction
            FROM predictions
            WHERE correct_prediction IS NOT NULL
            ORDER BY id
        """)
        records = cursor.fetchall()
        accuracy_over_time = []
        if records:
            cumulative_correct = 0
            for i, (_, correct) in enumerate(records, start=1):
                if correct == 1:
                    cumulative_correct += 1
                accuracy = round((cumulative_correct / i) * 100, 2)
                accuracy_over_time.append(accuracy)

        # --- Prediction Distribution ---
        cursor.execute("""
            SELECT predicted_class, COUNT(*) 
            FROM predictions
            GROUP BY predicted_class
        """)
        dist_data = cursor.fetchall()
        dist_labels = [row[0] for row in dist_data]
        dist_counts = [row[1] for row in dist_data]

        # --- Feedback Summary ---
        cursor.execute("""
            SELECT correct_prediction, COUNT(*)
            FROM predictions
            WHERE correct_prediction IS NOT NULL
            GROUP BY correct_prediction
        """)
        feedback_data = cursor.fetchall()
        feedback_labels = [
            "Correct" if row[0] == 1 else "Incorrect"
            for row in feedback_data
        ]
        feedback_counts = [row[1] for row in feedback_data]

    return render_template(
        'dashboard.html',
        accuracy_over_time=accuracy_over_time,
        dist_labels=dist_labels,
        dist_counts=dist_counts,
        feedback_labels=feedback_labels,
        feedback_counts=feedback_counts
    )

# -----------------------------------------------------------------------------
# Serve Uploaded Images
# -----------------------------------------------------------------------------
@app.route('/images/<filename>')
def send_image(filename):
    """Serves uploaded images."""
    return send_from_directory(UPLOAD_FOLDER, filename)

# -----------------------------------------------------------------------------
# Run Application
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Upload folder path:", UPLOAD_FOLDER)
    app.run(port=3000, debug=True)
