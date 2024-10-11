# import uvicorn
import sys

import fastapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as transforms
# sys.path.append("../Code")
# from Code.Models import Model
import Model
app = fastapi.FastAPI()

class ImageData(BaseModel):
    image: str

@app.get("/")
def hello():
    """ Main page of the app. """
    return "Hello World!"

@app.post("/predict")
async def predict_image(data: ImageData):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_data))

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((32,32)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Preprocess the image (resize, normalize, etc.)
        preprocess_image = transform(image)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Convert to PyTorch tensor and move to the appropriate device
        input_tensor = preprocess_image.clone()
        input_tensor = input_tensor.unsqueeze(0) # Add batch dimension
        input_tensor = input_tensor.to(device)

        model = Model.Net()
        try:
            # PATH = 'Models\Best_model.pt'
            PATH = 'Best_model.pt'
            model.load_state_dict(torch.load(PATH, weights_only=True))
        except:
            PATH = 'D:\PMLDL_ASS1\PMLDL_ASS1\Models\Best_model.pt'
            model.load_state_dict(torch.load(PATH, weights_only=True))
        model.to(device)

        model.eval()

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)

        # Get the predicted class (assuming a classification model)
        predicted_class = torch.argmax(outputs).item()

        predicted_class = model.Get_class_from_prediction(predicted_class)

        # Return the prediction
        return JSONResponse(
            content={"prediction": predicted_class},
            status_code=200
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

#if __name__ == '__main__':
    #uvicorn.run(app, host='127.0.0.1', port=8000)
