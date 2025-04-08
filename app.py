import torch
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify

from PIL import Image
from pymongo import MongoClient

app = Flask(__name__)


# MongoDB setup
client = MongoClient("mongodb+srv://username:password@cluster0.mongodb.net/test?retryWrites=true&w=majority")
db = client["skin_db"]
collection = db["disease_info"]
doctor_collection = db["doctors"]

# Load class labels
with open("class_labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load PyTorch model
model = models.resnet50(pretrained=False)
num_classes = len(classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet50_skin_disease.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image = Image.open(image.stream).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = classes[predicted_idx.item()]
        confidence_score = confidence.item()

    # Grade confidence
    if confidence_score < 0.5:
        grade = "Low"
    elif confidence_score < 0.75:
        grade = "Medium"
    else:
        grade = "High"

    # Fetch disease info
    disease_info = collection.find_one({"disease": predicted_class})
    if disease_info:
        disease_info.pop("_id", None)
    else:
        disease_info = {"description": "Info not found", "precautions": []}

    return jsonify({
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "grade": grade,
        "disease_info": disease_info
    })

@app.route("/find_doctors", methods=["GET"])
def find_doctors():
    specialty = request.args.get("specialty")
    location = request.args.get("location")

    query = {}
    if specialty:
        query["specialty"] = specialty
    if location:
        query["location"] = location

    doctors = list(doctor_collection.find(query, {"_id": 0}))
    return jsonify(doctors)

if __name__ == "__main__":
    app.run(debug=True)
