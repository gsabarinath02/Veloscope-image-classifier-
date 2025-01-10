import os
import torch
from torchvision import transforms
from PIL import Image
import timm

# ==========================
# 1. Load the Model
# ==========================
def load_model(model_path, device):
    """
    Loads the trained model for inference.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Initialize EfficientNet-B0 model
    model = timm.create_model('efficientnet_b0', pretrained=False)

    # Modify the classifier for binary classification
    num_features = model.get_classifier().in_features
    model.classifier = torch.nn.Linear(num_features, 2)

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Specify the path to the saved model
MODEL_PATH = 'efficientnet_b0_final.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = load_model(MODEL_PATH, device)
print("Model loaded successfully!")

# ==========================
# 2. Define Image Transformation
# ==========================
# The same transformations used during training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==========================
# 3. Predict Function for a Folder
# ==========================
def predict_folder(folder_path, model, transform, device):
    """
    Predicts the class (valid or invalid) for all images in a given folder.

    Args:
        folder_path (str): Path to the folder containing images.
        model (torch.nn.Module): Trained model.
        transform (torchvision.transforms.Compose): Transformations to apply to images.
        device (torch.device): Device to run the model on.

    Returns:
        List of tuples: Each tuple contains (image_name, prediction).
    """
    predictions = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Ensure it's a valid image file
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            continue

        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)

            # Append the prediction
            class_name = 'valid' if pred.item() == 0 else 'invalid'
            predictions.append((image_name, class_name))

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

    return predictions

# ==========================
# 4. Run Inference on a Folder
# ==========================
folder_path = '/images/invalid_images'  # Replace with the folder path containing images

# Get predictions for all images in the folder
predictions = predict_folder(folder_path, model, transform, device)

# Print the predictions
print("\nPredictions:")
for image_name, class_name in predictions:
    print(f"Image: {image_name}, Predicted Class: {class_name}")