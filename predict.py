import torch
from torchvision import transforms
from PIL import Image
import json
from model import get_model
import os

# Define solutions and deficiencies based on common plant diseases
SOLUTIONS = {
    # Tomato diseases
    'tomato_healthy': {'solution': 'No action needed. Plant is healthy.', 'deficiency': 'None'},
    'tomato_bacterial_spot': {'solution': 'Apply copper-based fungicide. Improve air circulation.', 'deficiency': 'None'},
    'tomato_early_blight': {'solution': 'Remove affected leaves. Apply fungicide. Mulch around plants.', 'deficiency': 'None'},
    'tomato_late_blight': {'solution': 'Remove and destroy affected plants. Apply fungicide.', 'deficiency': 'None'},
    'tomato_leaf_mold': {'solution': 'Improve ventilation. Apply fungicide.', 'deficiency': 'None'},
    'tomato_septoria_leaf_spot': {'solution': 'Remove affected leaves. Apply fungicide.', 'deficiency': 'None'},
    'tomato_spider_mites': {'solution': 'Spray with insecticidal soap. Increase humidity.', 'deficiency': 'None'},
    'tomato_target_spot': {'solution': 'Apply fungicide. Remove affected leaves.', 'deficiency': 'None'},
    'tomato_yellow_leaf_curl_virus': {'solution': 'Remove infected plants. Control whiteflies.', 'deficiency': 'None'},
    'tomato_mosaic_virus': {'solution': 'Remove infected plants. Disinfect tools.', 'deficiency': 'None'},
    'tomato_fusarium_wilt': {'solution': 'Remove infected plants. Improve soil drainage. Use resistant varieties.', 'deficiency': 'None'},
    'tomato_verticillium_wilt': {'solution': 'Remove infected plants. Rotate crops. Use resistant varieties.', 'deficiency': 'None'},
    'tomato_blossom_end_rot': {'solution': 'Ensure consistent watering. Apply calcium supplements.', 'deficiency': 'Calcium'},

    # Potato diseases
    'potato_healthy': {'solution': 'No action needed. Plant is healthy.', 'deficiency': 'None'},
    'potato_early_blight': {'solution': 'Apply fungicide. Remove affected leaves.', 'deficiency': 'None'},
    'potato_late_blight': {'solution': 'Remove affected plants. Apply fungicide.', 'deficiency': 'None'},
    'potato_scab': {'solution': 'Adjust soil pH. Use resistant varieties.', 'deficiency': 'None'},
    'potato_blackleg': {'solution': 'Remove infected plants. Disinfect tools.', 'deficiency': 'None'},
    'potato_virus_y': {'solution': 'Remove infected plants. Control aphids.', 'deficiency': 'None'},

    # Corn diseases
    'corn_healthy': {'solution': 'No action needed. Plant is healthy.', 'deficiency': 'None'},
    'corn_common_rust': {'solution': 'Apply fungicide. Plant resistant varieties.', 'deficiency': 'None'},
    'corn_gray_leaf_spot': {'solution': 'Apply fungicide. Rotate crops.', 'deficiency': 'None'},
    'corn_northern_leaf_blight': {'solution': 'Apply fungicide. Plant resistant varieties.', 'deficiency': 'None'},
    'corn_southern_rust': {'solution': 'Apply fungicide. Improve field drainage.', 'deficiency': 'None'},
    'corn_bacterial_leaf_streak': {'solution': 'Use resistant varieties. Avoid overhead irrigation.', 'deficiency': 'None'},

    # Other plants
    'apple_healthy': {'solution': 'No action needed. Plant is healthy.', 'deficiency': 'None'},
    'apple_scab': {'solution': 'Apply fungicide. Prune affected branches.', 'deficiency': 'None'},
    'apple_cedar_apple_rust': {'solution': 'Apply fungicide. Remove nearby cedar trees.', 'deficiency': 'None'},
    'apple_black_rot': {'solution': 'Remove infected fruit and branches. Apply fungicide.', 'deficiency': 'None'},

    'grape_healthy': {'solution': 'No action needed. Plant is healthy.', 'deficiency': 'None'},
    'grape_black_rot': {'solution': 'Apply fungicide. Remove infected parts.', 'deficiency': 'None'},
    'grape_downy_mildew': {'solution': 'Improve air circulation. Apply fungicide.', 'deficiency': 'None'},
    'grape_powdery_mildew': {'solution': 'Apply sulfur-based fungicide. Improve ventilation.', 'deficiency': 'None'},

    'peach_healthy': {'solution': 'No action needed. Plant is healthy.', 'deficiency': 'None'},
    'peach_bacterial_spot': {'solution': 'Apply copper fungicide. Avoid overhead watering.', 'deficiency': 'None'},
    'peach_peach_leaf_curl': {'solution': 'Apply fungicide before bud break. Prune affected leaves.', 'deficiency': 'None'},

    # Nutrient deficiencies
    'nitrogen_deficiency': {'solution': 'Apply nitrogen-rich fertilizer like urea or ammonium nitrate.', 'deficiency': 'Nitrogen'},
    'phosphorus_deficiency': {'solution': 'Apply phosphorus-rich fertilizer like superphosphate.', 'deficiency': 'Phosphorus'},
    'potassium_deficiency': {'solution': 'Apply potassium-rich fertilizer like potassium chloride.', 'deficiency': 'Potassium'},
    'calcium_deficiency': {'solution': 'Apply calcium-rich amendments like gypsum or lime.', 'deficiency': 'Calcium'},
    'magnesium_deficiency': {'solution': 'Apply magnesium-rich fertilizer like Epsom salt.', 'deficiency': 'Magnesium'},
    'iron_deficiency': {'solution': 'Apply iron chelate. Adjust soil pH to 6.0-7.0.', 'deficiency': 'Iron'},
    'zinc_deficiency': {'solution': 'Apply zinc sulfate. Ensure soil pH is not too high.', 'deficiency': 'Zinc'},
    'manganese_deficiency': {'solution': 'Apply manganese sulfate. Adjust soil pH.', 'deficiency': 'Manganese'},
    'boron_deficiency': {'solution': 'Apply borax. Test soil boron levels.', 'deficiency': 'Boron'},
    'copper_deficiency': {'solution': 'Apply copper sulfate. Avoid over-fertilization.', 'deficiency': 'Copper'},
    'sulfur_deficiency': {'solution': 'Apply sulfur-containing fertilizers.', 'deficiency': 'Sulfur'},
}

def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_disease(image_path, model, classes, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score

def get_solution_and_deficiency(predicted_class):
    if predicted_class in SOLUTIONS:
        return SOLUTIONS[predicted_class]['solution'], SOLUTIONS[predicted_class]['deficiency']
    else:
        return "Unknown condition. Consult an expert.", "Unknown"

def predict_and_output_json(image_path, model_path='plant_model.pth', classes_path='classes.json'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        return json.dumps({"error": "Model file not found. Please train the model first."})

    if not os.path.exists(classes_path):
        return json.dumps({"error": "Classes file not found. Please train the model first."})

    with open(classes_path, 'r') as f:
        classes = json.load(f)

    num_classes = len(classes)
    model = load_model(model_path, num_classes).to(device)

    predicted_class, confidence = predict_disease(image_path, model, classes, device)
    solution, deficiency = get_solution_and_deficiency(predicted_class)

    # Extract plant type and condition
    parts = predicted_class.split('_', 1)
    plant_type = parts[0] if len(parts) > 1 else 'Unknown'
    condition = parts[1] if len(parts) > 1 else predicted_class

    result = {
        "plant_type": plant_type,
        "predicted_disease_or_condition": condition,
        "confidence": round(confidence * 100, 2),
        "deficiency": deficiency,
        "recommended_solution": solution,
        "stats": {
            "model_used": "EfficientNet-B1-based CNN",
            "input_image": image_path,
            "prediction_time": "N/A"  # Could add timing if needed
        }
    }

    return json.dumps(result, indent=2)

if __name__ == '__main__':
    # Example usage
    image_path = 'sample_leaf.jpg'  # Replace with actual image path
    if os.path.exists(image_path):
        output = predict_and_output_json(image_path)
        print(output)
    else:
        print("Sample image not found. Please provide a valid image path.")
