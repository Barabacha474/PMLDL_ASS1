# CIFAR-100 Image Classifier
## Project info

This is study pet project, consisting of CIFAR-100 trained image classifier, with FastAPI endpoint and streamlit interface.

## Requirements

**Python 3.9** 

All necessary libraries stated in requirements.txt, you can use it via ```pip install -r requirements.txt```

## How to run
1. Run **Code/Deployment/api/main.py** to check for existence of pretrained model, train model if needed (cuda compatible) and run FastAPI endpoint to access model with JsonBase64 images (32x32 pixels)  
2. After that, you also can run **Code/Deployment/app/main.py** via **terminal** to run Streamlit app for simpler interface and ability to upload jpg, jpeg and png images to model.

You can find example images, both in JsonBase64 and jpg in **TestImages**