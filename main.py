import torch
from flask import Flask, request, jsonify
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

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400
    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0).to(device)
    output = model(img)
    _, predicted = torch.max(output, 1)
    return jsonify({'digits': predicted.item()}) #MNIST numbers

if __name__ == '__main__':
    app.run(debug=True)
