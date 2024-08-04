import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from models import HCRM, HSRM
from torchvision import transforms
import io
import torch.nn.functional as F
import matplotlib.pyplot as plt

### FOR INFERENCE LATER ###

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_single = HCRM().to(device)
model_single.load_state_dict(torch.load('./model/handwritten_character_recognition_model.pth'))
model_sequence = HSRM().to(device)
model_sequence.load_state_dict(torch.load('./model/handwritten_character_recognition_model_lstm.pth'))

model_single.eval()
model_sequence.eval()

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_sequence_image(image, num_digits=5):
    width, height = image.size
    print("Width:", width, "Height:", height)
    digit_width = width // num_digits
    images = []
    for i in range(num_digits):
        digit = image.crop((i * digit_width, 0, (i + 1) * digit_width, height))
        digit = transform(digit)
        #plt.imshow(digit) # DEBUGGING
        images.append(digit)
    return torch.stack(images).unsqueeze(0) # shape(1, num_digits, 1, 28, 28)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('L')

    model_choice = request.form.get('model_choice')
    if model_choice == 'single':
        img = transform(img).unsqueeze(0).to(device)
        outputs = model_single(img)
        print("outputs:", outputs)
        _, predicted = torch.max(outputs.data, 1)
        result = predicted.squeeze().tolist()
        print("result:", result)
    else:
        img = preprocess_sequence_image(img).to(device)
        outputs = model_sequence(img)
        print("outputs:", outputs)
        _, predicted = torch.max(outputs.data, 2)
        print("predicted:", predicted)
        result = predicted.squeeze().tolist()
        print("result:", result)

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)