import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pydicom as dicom
from keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model("model.h5")

# Define a function to preprocess the image before making predictions
def preprocess_image(image_path):
    img = dicom.read_file(image_path).pixel_array
    img = cv2.resize(img, (256, 256))
    img = img.reshape(1, 256, 256, 1)
    return img

# Define a function to make predictions on the preprocessed image
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction

# Define a function to handle the button click event
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path)
        alert_result(prediction)

# Define a function to display the prediction result as an alert
def alert_result(prediction):
    # In this example, we assume the model predicts two classes (binary classification)
    if prediction[0, 0] > prediction[0, 1]:
        result_text = "No Cancer"
    else:
        result_text = "Cancer"

    result_label.config(text="Prediction: " + result_text)

# Create the main application window
app = tk.Tk()
app.title("Cancer Detection")

# Create a button to browse and select an image

title_label = tk.Label(app,text="Select the Image File")

browse_button = tk.Button(app, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Create a label to display the prediction result
result_label = tk.Label(app, text="")
result_label.pack(pady=5)

# Run the main event loop
app.mainloop()
