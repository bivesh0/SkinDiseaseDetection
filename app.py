import os
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# MongoDB setup
MONGO_URI = "mongodb+srv://thakur:thakur@diseasedescandprev.hxwp7.mongodb.net/?retryWrites=true&w=majority&appName=DiseaseDescandPrev"
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in environment variables!")

try:
    client = MongoClient(MONGO_URI)
    db = client["skin_disease_db"]
    disease_collection = db["disease_info"]
    doctor_collection = db["doctors"]
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

# Upload folder config
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
device = torch.device("cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 23)
model_path = "./resnet50_dermnet.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels
class_names = [
    'Acne', 'Alopecia', 'Bullous Disease', 'Dermatitis', 'Drug Eruptions',
    'Eczema', 'Impetigo', 'Lichen Planus', 'Lupus', 'Malignant Lesions',
    'Nail Fungus', 'Nail Psoriasis', 'Psoriasis', 'Rosacea', 'Scabies',
    'Seborrheic Keratoses Tumors', 'Systemic Disease', 'Tinea Ringworm',
    'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Vitiligo', 'Warts'
]

def get_confidence_grade(confidence):
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.50:
        return "Mid"
    else:
        return "Low"

def get_disease_info(disease_name):
    result = disease_collection.find_one({"name": disease_name}, {"_id": 0})
    return result or {"description": "No details available.", "prevention": "No preventive measures available."}

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        confidence_grade = get_confidence_grade(confidence_score)
        disease_info = get_disease_info(predicted_class)
        return predicted_class, confidence_score, confidence_grade, disease_info
    except Exception as e:
        return "Error", 0.0, "Low", {"description": "Prediction failed.", "prevention": str(e)}

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file selected"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        prediction, score, grade, info = predict_image(filepath)

        return render_template("index.html",
                               prediction=prediction,
                               confidence_score=f"{score:.2f}",
                               confidence_grade=grade,
                               disease_info=info,
                               filename=filename)

    return render_template("index.html", prediction=None, confidence_score=None,
                           confidence_grade=None, disease_info=None, filename=None)

def get_filtered_doctors(location=None, specialty=None):
    query = {}
    if location and location.lower() != "all":
        query["location"] = {"$regex": location, "$options": "i"}
    if specialty and specialty.lower() != "all":
        query["specialty"] = {"$regex": specialty, "$options": "i"}
    return list(doctor_collection.find(query, {"_id": 0}))

@app.route("/doctors", methods=["GET"])
def doctors():
    location = request.args.get("location")
    specialty = request.args.get("specialty")

    doctors_list = get_filtered_doctors(location, specialty)

    unique_locations = [
        "Peddapuram", "Tadepalligudem", "Kakinada Road", "Unduru",
        "Rajahmundry", "Rajamahendrvaram", "Danavaipeta"
    ]
    return render_template("doctors.html",
                           doctors=doctors_list,
                           unique_locations=unique_locations,
                           unique_specialties=class_names)

if __name__ == "__main__":
    app.run(debug=False, host="localhost", port=5000)
