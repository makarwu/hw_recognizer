import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from models import HCRM
from torchvision import transforms
import io

### FOR INFERENCE LATER ###

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HCRM().to(device)
model.load_state_dict(torch.load('./model/handwritten_character_recognition_model.pth'))
model.eval()

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        
        return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
