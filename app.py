import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from pymongo import MongoClient  # MongoDB client
from PIL import Image
from torchvision import models

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
MONGO_URI = os.getenv("MONGO_URI")  # Fetch MongoDB URI from Render env variables
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in environment variables!")
try:
    client = MongoClient(MONGO_URI)
    db = client["skin_disease_db"]  # Database name
    collection = db["disease_info"]  # Collection name
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

# Ensure 'static/uploads' directory exists (TEMPORARY STORAGE ONLY)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 23)  # Adjust output layer for 23 classes
model_path = "./resnet50_dermnet.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found! Ensure it's included in your Render repo.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Acne', 'Alopecia', 'Bullous Disease', 'Dermatitis', 'Drug Eruptions', 
               'Eczema', 'Impetigo', 'Lichen Planus', 'Lupus', 'Malignant Lesions', 
               'Nail Fungus', 'Nail Psoriasis', 'Psoriasis', 'Rosacea', 'Scabies', 
               'Seborrheic Keratoses Tumors', 'Systemic Disease', 'Tinea Ringworm',
               'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Vitiligo', 'Warts']

# Confidence grading function
def get_confidence_grade(confidence):
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.50:
        return "Mid"
    else:
        return "Low"

# Function to fetch disease details from MongoDB
def get_disease_info(disease_name):
    result = collection.find_one({"name": disease_name}, {"_id": 0})  # Exclude MongoDB ID
    return result if result else {"description": "No details available.", "prevention": "No preventive measures available."}

# Function to make a prediction with confidence score
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)  # Get highest confidence

        predicted_class = class_names[predicted_class_idx.item()]
        confidence_score = confidence.item()
        confidence_grade = get_confidence_grade(confidence_score)

        # Fetch disease details from MongoDB
        disease_info = get_disease_info(predicted_class)

        return predicted_class, confidence_score, confidence_grade, disease_info
    except Exception as e:
        return "Error", 0.0, "Low", {"description": "Prediction failed.", "prevention": str(e)}

# Flask route to handle uploads and predictions
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            predicted_class, confidence_score, confidence_grade, disease_info = predict_image(filepath)

            return render_template("index.html", 
                                   prediction=predicted_class, 
                                   confidence_score=f"{confidence_score:.2f}", 
                                   confidence_grade=confidence_grade,
                                   disease_info=disease_info,
                                   filename=filename)

    return render_template("index.html", prediction=None, confidence_score=None, confidence_grade=None, disease_info=None, filename=None)

# Function to filter doctors from MongoDB
def get_filtered_doctors(location=None, specialty=None):
    query = {}

    if location and location.lower() != "all":
        query["location"] = {"$regex": location, "$options": "i"}  # Case-insensitive match

    if specialty and specialty.lower() != "all":
        query["specialty"] = {"$regex": specialty, "$options": "i"}  # Case-insensitive match

    # Fetch filtered doctors directly from MongoDB
    doctors_list = list(db["doctors"].find(query, {"_id": 0}))

    return doctors_list

# Route for doctors page
@app.route("/doctors", methods=["GET"])
def doctors():
    location = request.args.get("location")
    specialty = request.args.get("specialty")

    doctors_list = get_filtered_doctors(location, specialty)
    
    unique_locations = [
        "Peddapuram", "Tadepalligudem", "Kakinada Road", "Unduru", "Rajahmundry",
        "Rajamahendrvaram", "Danavaipeta"
    ]

    unique_specialties = class_names  # Use disease class names as specialties

    return render_template("doctors.html", 
                           doctors=doctors_list, 
                           unique_locations=unique_locations, 
                           unique_specialties=unique_specialties)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
