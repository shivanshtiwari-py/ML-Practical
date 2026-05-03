import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'model/tumor_model.h5'
UPLOAD_FOLDER = 'static/uploads'
# Update these to match your specific model requirements
IMG_SIZE = (256, 256) 
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary'] # Replace with your actual labels

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your model once when the server starts
model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization
    
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions[0])
    return CLASSES[result_index], float(np.max(predictions[0]))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            label, confidence = predict_image(filepath)
            return render_template('index.html', 
                                 label=label, 
                                 confidence=f"{confidence*100:.2f}%",
                                 user_image=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)