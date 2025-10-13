import os

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('model.h5')

class_labels = ['covid', 'normal', 'pneumonia', 'tuberculosis']

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():

    if not os.path.exists('./images'):
        os.makedirs('./images')

    imagefile = request.files['imagefile']
    image_path = os.path.join('./images', imagefile.filename)
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize pixel values

    yhat  = model.predict(img_array)
    predicted_class = np.argmax(yhat, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = np.max(yhat) * 100

    classification = f"{predicted_label} ({confidence:.2f}%)"

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)

