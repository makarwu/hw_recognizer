import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from models import HCRM, HSRM
from torchvision import transforms
import io
import torch.nn.functional as F

### FOR INFERENCE LATER ###

sequence_length = 5

# Transforms for single digit images
transform_single = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Transforms for sequence images
transform_sequence = transforms.Compose([
    transforms.Resize((28 * sequence_length, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_sequence_image(image, sequence_length=sequence_length):
    # Ensure the image size matches the expected sequence length, assume each digit is 28x28
    width, height = image.size
    print("width:", width, "height:", height)
    assert width == 28 * sequence_length and height == 28, "Image size does not match the expected dimensions."

    # Split the image into individual digit images
    digit_images = [image.crop((i * 28, 0, (i + 1) * 28, 28)) for i in range(sequence_length)]
    
    # Apply the same transforms to each digit image and stack them
    digit_tensors = [transform_single(digit_image).unsqueeze(0) for digit_image in digit_images]
    
    # Stack into a single tensor with shape (sequence_length, 1, 28, 28)
    sequence_tensor = torch.cat(digit_tensors, dim=0)
    
    return sequence_tensor.unsqueeze(0)  # Add batch dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_single = HCRM().to(device)
model_single.load_state_dict(torch.load('./model/handwritten_character_recognition_model.pth'))
model_sequence = HSRM().to(device)
model_sequence.load_state_dict(torch.load('./model/handwritten_character_recognition_model_lstm_2.pth'))

model_single.eval()
model_sequence.eval()

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            model_choice = request.form['model']

            if model_choice == 'single':
                img = transform_single(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model_single(img)
                    _, predicted = torch.max(output, 1)
                    prediction = str(predicted.item())
            else:
                img = transform_sequence(img)  # Ensure it's grayscale
                sequence_tensor = preprocess_sequence_image(img).to(device)
                with torch.no_grad():
                    output = model_sequence(sequence_tensor)
                    print(output)  # Debugging statement
                    print(output.shape)  # Debugging statement
                    _, predicted = torch.max(output, 2)
                    prediction = ''.join([str(digit.item()) for digit in predicted.squeeze()])

            return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)