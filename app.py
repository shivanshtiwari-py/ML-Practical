import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Use the current directory for everything
app = Flask(__name__, template_folder='.')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'tumor_model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model(MODEL_PATH)

def predict_label(img_path):
    # CRITICAL: Ensure target_size matches your model's training!
    img = image.load_img(img_path, target_size=(256, 256)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary'] # Replace with your actual labels
    return classes[np.argmax(prediction)], f"{np.max(prediction)*100:.2f}%"

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        try:
            img = request.files.get('file')
            if img:
                img_path = os.path.join(UPLOAD_FOLDER, img.filename)
                img.save(img_path)
                
                label, confidence = predict_label(img_path)
                return render_template("index.html", prediction=label, confidence=confidence)
        except Exception as e:
            # This will show the error on the webpage instead of vanishing
            return f"Error during prediction: {str(e)}"
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
