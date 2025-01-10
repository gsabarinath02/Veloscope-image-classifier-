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
# 3. Prediction Function
# ==========================
def predict_image(image_path, model, transform, device):
    """
    Predicts whether an image is 'valid' or 'invalid'.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): Trained model.
        transform (torchvision.transforms.Compose): Transformations to apply to the image.
        device (torch.device): Device to run the model on.

    Returns:
        str: 'valid' or 'invalid' based on prediction.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)

        # Return the class name
        return 'valid' if pred.item() == 0 else 'invalid'

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# ==========================
# 4. Test the Prediction
# ==========================
image_path = '/images/Valid_images/146_0091e3b5d5d3a595bea050b45a3a55512d5a6019cc5b9e50_7_P_DSC00723.JPG'  # Replace with the path to your image

# Get the prediction
prediction = predict_image(image_path, model, transform, device)

# Print the result
if prediction is not None:
    print(f"Prediction for the image '{image_path}': {prediction}")
