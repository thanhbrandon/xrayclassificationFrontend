import os
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print("Upload folder path:", UPLOAD_FOLDER)

model = load_model('model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']

    if imagefile.filename == '':
        return render_template('index.html', error="No file selected")

    image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
    imagefile.save(image_path)

    # Preprocess and predict
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    classes = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
    predicted_class = classes[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)

    return render_template('index.html',
                           filename=imagefile.filename,
                           prediction=f"{predicted_class} ({confidence}%)")

@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
