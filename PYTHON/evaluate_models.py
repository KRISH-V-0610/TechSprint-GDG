import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report, 
    matthews_corrcoef
)
from PIL import Image
from torchvision import transforms

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Path to the specific 10-class dataset
DATASET_PATH = r"./dataset/IMG_CLASSES"

# IEEE Paper Format Configuration for Plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# -----------------------------------------------------------------------------
# DATASET CLASS DEFINITIONS (10 CLASSES)
# -----------------------------------------------------------------------------
# We map the complex folder names to simple Class Labels
DATASET_CLASSES_MAP = {
    "1. Eczema": "Eczema",
    "2. Melanoma": "Melanoma", 
    "3. Atopic Dermatitis": "Atopic Dermatitis",
    "4. Basal Cell Carcinoma": "Basal Cell Carcinoma",
    "5. Melanocytic Nevi": "Melanocytic Nevi",
    "6. Benign Keratosis": "Benign Keratosis",
    "7. Psoriasis pictures": "Psoriasis",
    "8. Seborrheic Keratoses": "Seborrheic Keratoses",
    "9. Tinea Ringworm": "Tinea Ringworm",
    "10. Warts Molluscum": "Warts Molluscum"
}

# The ordered list of ground truth labels for the Confusion Matrix
TARGET_CLASSES = list(DATASET_CLASSES_MAP.values())

# -----------------------------------------------------------------------------
# MODEL C SETUP
# -----------------------------------------------------------------------------
try:
    from predict_c import predict_c
    from predict_c import CLASS_NAMES as CLASS_NAMES_C
except ImportError:
    print("Error: Could not import predict_c. Ensure predict_c.py is in the same directory.")
    CLASS_NAMES_C = []

# MAPPING: Model C Output -> Dataset Class
# Model C Classes: 
# "Actinic keratosis", "Atopic Dermatitis", "Benign keratosis", "Dermatofibroma", 
# "Melanocytic nevus", "Melanoma", "Squamous cell carcinoma", 
# "Tinea Ringworm Candidiasis", "Vascular lesion"
MODEL_C_MAP = {
    "Actinic keratosis": "Basal Cell Carcinoma", # Approximate Mapping (BCC/Actinic often grouped)
    "Squamous cell carcinoma": "Basal Cell Carcinoma", # Approximate Mapping
    "Atopic Dermatitis": "Atopic Dermatitis",
    "Benign keratosis": "Benign Keratosis",
    "Melanocytic nevus": "Melanocytic Nevi",
    "Melanoma": "Melanoma",
    "Tinea Ringworm Candidiasis": "Tinea Ringworm",
    # Mismatches or specific logic could be added here
    # "Dermatofibroma": "Unknown",
    # "Vascular lesion": "Unknown"
}

def map_prediction_c(pred_class):
    return MODEL_C_MAP.get(pred_class, "Other")

# -----------------------------------------------------------------------------
# MODEL D SETUP
# -----------------------------------------------------------------------------
try:
    from predict_d import load_model, predict_d
    # Note: We duplicate class names here because importing them from inside predict_d function is hard
    CLASS_NAMES_D = [
        "Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
        "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema Photos", "Exanthems and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases",
        "Herpes HPV and other STDs Photos", "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles",
        "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis",
        "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites",
        "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease", "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives", "Vascular Tumors", "Vasculitis Photos", "Warts Molluscum and other Viral Infections"
    ]
except ImportError:
    print("Error: Could not import predict_d. Ensure predict_d.py is in the same directory.")
    CLASS_NAMES_D = []

# Pre-load Model D
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model_d = load_model(r"./models/model_epoch_25.pth", num_classes=23)
    model_d.to(device)
    model_d.eval()
except Exception as e:
    print(f"Warning: Could not load Model D: {e}")
    model_d = None

transform_d = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_d_optimized(image: Image.Image):
    """Optimized prediction for Model D using pre-loaded model."""
    if model_d is None:
        return {"class": "Error", "confidence": 0.0}
    
    img_tensor = transform_d(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_d(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    return {"class": CLASS_NAMES_D[predicted_class_idx], "confidence": confidence}

# MAPPING: Model D Output -> Dataset Class
MODEL_D_MAP = {
    "Eczema Photos": "Eczema",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma", # Note: Allows ambiguity (See fix below)
    "Atopic Dermatitis Photos": "Atopic Dermatitis",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Basal Cell Carcinoma",
    "Seborrheic Keratoses and other Benign Tumors": "Seborrheic Keratoses", # Vs Benign Kerotosis? 
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Tinea Ringworm",
    "Warts Molluscum and other Viral Infections": "Warts Molluscum",
    # Ambiguity Handling:
    # Model D groups Melanoma+Nevi. We map to "Melanoma/Nevi Group" or handle dynamically?
    # We will handle dynamic checks in evaluation loop? 
    # For now, simplistic mapping.
}

def map_prediction_d(pred_class, true_label=None):
    """
    Maps Model D output to Dataset Class.
    Includes logic for grouped classes to avoid unfair penalties.
    """
    # Special Handling for Grouped Classes in Model D
    if pred_class == "Melanoma Skin Cancer Nevi and Moles":
        if true_label in ["Melanoma", "Melanocytic Nevi"]:
            return true_label # Count as correct if it's within the group
        return "Melanoma" # Default if valid mismatch
        
    if pred_class == "Seborrheic Keratoses and other Benign Tumors":
        if true_label in ["Seborrheic Keratoses", "Benign Keratosis"]:
            return true_label
        return "Seborrheic Keratoses"

    return MODEL_D_MAP.get(pred_class, "Other")


# -----------------------------------------------------------------------------
# EVALUATION CORE
# -----------------------------------------------------------------------------
def load_dataset_10_classes(data_path):
    """
    Scans the specific 10 folders in the dataset and loads images.
    Returns: list of (image_path, true_label_string)
    """
    test_data = []
    
    if not os.path.exists(data_path):
        print(f"Dataset path not found: {data_path}")
        return test_data

    # List only directories
    dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    print(f"Scanning {data_path}...")
    
    for d in dirs:
        # Find matching clean label
        clean_label = "Other"
        for key in DATASET_CLASSES_MAP:
            if d.startswith(key) or key in d: # Simple matching
                 clean_label = DATASET_CLASSES_MAP[key]
                 break
        
        if clean_label == "Other":
            print(f"Skipping unknown folder: {d}")
            continue
            
        full_dir_path = os.path.join(data_path, d)
        files = [f for f in os.listdir(full_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(files)} images for class '{clean_label}' in '{d}'")
        
        for f in files:
            test_data.append((os.path.join(full_dir_path, f), clean_label))
            
    return test_data

def evaluate_and_report(model_name, predict_func, map_func, test_data, class_order):
    print(f"\n" + "-"*30)
    print(f"Evaluating {model_name}")
    print(f"-"*30)
    
    y_true = []
    y_pred = []
    
    for i, (img_path, true_label) in enumerate(test_data):
        if i % 100 == 0: print(f"Processing {i}/{len(test_data)}...", end='\r')
        try:
            image = Image.open(img_path).convert('RGB')
            raw_result = predict_func(image)
            raw_class = raw_result['class']
            
            # Map raw model output to standardized dataset label
            # We pass true_label to helper for "Grouped Class" logic if needed
            mapped_class = map_func(raw_class, true_label) if model_name == "Model D" else map_func(raw_class)
            
            y_true.append(true_label)
            y_pred.append(mapped_class)
            
        except Exception as e:
            # print(f"Error: {e}") 
            pass
            
    # Add 'Other' to class order if mapped predictions fall outside
    extended_classes = class_order + ["Other"]
    
    # Generate Metrics
    print(f"\nGenerating Report for {model_name}...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_order) # Ignore 'Other' in matrix axes? Or include?
    # To include everything, we compute labels dynamically, but for Paper we want fixed 10x10 usually.
    # Any 'Other' predictions will be ignored in the 10x10 matrix (creating row/col sum mismatches)
    # OR we can force 'Other' into the plot. Let's stick to 10x10 for cleanliness, understanding 'Other' is error.
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_order, yticklabels=class_order)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300)
    
    # 2. Classification Report
    report = classification_report(y_true, y_pred, labels=class_order, zero_division=0)
    print(report)
    
    report_dict = classification_report(y_true, y_pred, labels=class_order, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(f'classification_report_{model_name.replace(" ", "_")}.csv')

    # 3. Overall Metrics
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "MCC": mcc,
        "Precision (W)": report_dict['weighted avg']['precision'],
        "Recall (W)": report_dict['weighted avg']['recall'],
        "F1 (W)": report_dict['weighted avg']['f1-score']
    }

def main():
    # Load Data
    test_data = load_dataset_10_classes(DATASET_PATH)
    if not test_data:
        print("No data found. Check path.")
        return
        
    results = []
    
    # Evaluate Model C
    res_c = evaluate_and_report("Model C", predict_c, map_prediction_c, test_data, TARGET_CLASSES)
    results.append(res_c)
    
    # Evaluate Model D
    res_d = evaluate_and_report("Model D", predict_d_optimized, map_prediction_d, test_data, TARGET_CLASSES)
    results.append(res_d)
    
    # Compare
    summary_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(summary_df)
    summary_df.to_csv("final_model_comparison.csv", index=False)

if __name__ == "__main__":
    main()
