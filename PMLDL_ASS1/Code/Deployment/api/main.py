# import uvicorn
# import sys
import subprocess

import fastapi
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as transforms
import Model
app = fastapi.FastAPI()

import sys
import os
CURRENT_FILE_PATH = os.path.abspath(__file__)
CODE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
MODELS_DIR_PATH = os.path.join(CODE_DIR_PATH, "Models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR_PATH, "Best_model.pt")

os.makedirs(MODELS_DIR_PATH, exist_ok=True)


class ImageData(BaseModel):
    image: str


def train_model():
    """Run the training script from Model.py"""
    print("\n" + "=" * 50)
    print("Starting model training...")
    print("This may take several minutes to complete.")
    print("=" * 50 + "\n")

    try:
        # Run Model.py as a subprocess
        result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "Model.py")],
                                capture_output=False,  # Set to True if you want to capture output
                                text=False,
                                check=True)

        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("=" * 50 + "\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")
        return False
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        return False


def check_and_train_if_needed():
    """Check if model exists, ask user if training is needed"""
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found pre-trained model at: {MODEL_SAVE_PATH}")
        return True

    print("\n" + "=" * 50)
    print("WARNING: Pre-trained model not found!")
    print(f"Expected model at: {MODEL_SAVE_PATH}")
    print("=" * 50 + "\n")

    while True:
        response = input("Would you like to train the model now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            success = train_model()
            if success and os.path.exists(MODEL_SAVE_PATH):
                return True
            else:
                print("Training failed or model was not created.")
                return False
        elif response in ['n', 'no']:
            print("Exiting. Model is required for the API to function.")
            return False
        else:
            print("Please enter 'y' or 'n'.")


# Global model instance
model = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model when the app starts"""
    global model

    print("\nInitializing FastAPI application...")

    # Check if model exists and train if needed
    if not check_and_train_if_needed():
        print("Shutting down FastAPI application...")
        sys.exit(1)

    try:
        # Load the trained model
        print(f"Loading model from: {MODEL_SAVE_PATH}")

        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Initialize model
        model = Model.Net()

        # Load state dict with appropriate device mapping
        if device.type == 'cuda':
            model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        else:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))

        model.to(device)
        model.eval()  # Set to evaluation mode

        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Shutting down FastAPI application...")
        sys.exit(1)


@app.get("/")
def hello():
    """Main page of the app."""
    return {"message": "CIFAR-100 Image Classification API, try /docs to access model predictions manually!",
            "status": "Model is ready for predictions"}


@app.post("/predict")
async def predict_image(data: ImageData):
    try:
        if model is None:
            return JSONResponse(
                content={"error": "Model not loaded. Please restart the server."},
                status_code=500
            )

        # Decode the base64 image
        image_data = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_data))

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        # Preprocess the image
        preprocess_image = transform(image)

        # Convert to PyTorch tensor
        input_tensor = preprocess_image.unsqueeze(0)  # Add batch dimension

        # Move to the same device as model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)

        # Get the predicted class
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()

        # Get class name using the method from Model
        predicted_class = model.Get_class_from_prediction(predicted_idx)

        # Return the prediction
        return JSONResponse(
            content={
                "prediction": predicted_class,
                "class_id": predicted_idx,
                "confidence": torch.softmax(outputs, dim=1)[0][predicted_idx].item()
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == '__main__':
    if not check_and_train_if_needed():
        sys.exit(1)

    print("\nStarting FastAPI server...")
    print("Server will be available at: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server\n")

    uvicorn.run(app, host='127.0.0.1', port=8000)
