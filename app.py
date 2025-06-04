from flask import Flask, request, render_template
import torch
import os
from torchvision import transforms
from PIL import Image

# Import all model loaders
from model_cnn import get_model as get_cnn
from model_alexnet import get_model as get_alexnet
from model_vgg import get_model as get_vgg
from model_resnet import get_model as get_resnet
from model_lstm import get_model as get_lstm
from model_rcnn import get_model as get_rcnn

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model_map = {
    "CNN": ("models/cnn.pth", get_cnn),
    "AlexNet": ("models/alexnet.pth", get_alexnet),
    "VGG": ("models/vgg.pth", get_vgg),
    "ResNet": ("models/resnet.pth", get_resnet),
    "LSTM": ("models/lstm.pth", get_lstm),
    "RCNN": ("models/rcnn.pth", get_rcnn),
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        model_name = request.form.get("model")
        optimizer_name = request.form.get("optimizer")
        image_file = request.files.get("image")

        if model_name not in model_map:
            prediction = "Invalid model selected!"
            return render_template("index.html", prediction=prediction)

        if image_file:
            image_path = os.path.join("static", image_file.filename)
            image_file.save(image_path)

            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            model_path, get_model_fn = model_map[model_name]
            model = get_model_fn().to(device)

            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    output = model(image)
                    predicted_class = torch.argmax(output).item()
                    prediction = class_names[predicted_class] if 0 <= predicted_class < len(class_names) else "Unknown Class"
            else:
                prediction = "Model weights not found!"

    return render_template("index.html", prediction=prediction, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
