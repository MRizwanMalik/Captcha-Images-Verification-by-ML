from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io




def preprocess_image(image, img_width, img_height):
    image_resized = cv2.resize(image, (img_width, img_height))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    morph_img = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated_img = cv2.dilate(morph_img, kernel, iterations=1)
    edges = cv2.Canny(dilated_img, 50, 150)

    return edges / 255.0


def decode_prediction(pred, num_classes):
    pred_decoded = np.argmax(pred, axis=-1)
    if pred_decoded.ndim > 1:
        pred_decoded = pred_decoded.flatten()

    return ''.join([str(p) for p in pred_decoded if p < num_classes])


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

def test_single_image(interpreter, img_path, img_width, img_height, num_classes):
    original_image = cv2.imread(img_path)
    preprocessed_image = preprocess_image(original_image, img_width, img_height)

    X_test = np.array(preprocessed_image).reshape(img_width, img_height, 1)
    X_test = np.expand_dims(X_test, axis=0)

    prediction = tflite_predict(interpreter, X_test.astype(np.float32))
    predicted_label = decode_prediction(prediction, num_classes)
    return predicted_label

import os
# model_path = os.path.join(os.path.dirname(__file__), "Capmodel_098613.tflite")
interpreter = tf.lite.Interpreter(model_path=r"Capmodel_098613.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

@app.route('/ocr', methods=['GET', 'POST'])
def extract_text():
    if request.json['images'] != '':
        images = request.json['images']
        results = []
        try:
            for image in images:

                try:
                    # Decode base64 image, remove the prefix if necessary
                    image_base64 = image['image'].split(',')[1] if ',' in image['image'] else image['image']
                    img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
                    img_np = np.array(img)

                    # Preprocess the image to match the input size expected by the model
                    img_width, img_height = 150, 80  # Define width and height for model
                    preprocessed_img = preprocess_image(img_np, img_width, img_height)

                    # Prepare the image for prediction
                    X_test = np.array(preprocessed_img).reshape(img_width, img_height, 1)
                    X_test = np.expand_dims(X_test, axis=0)

                    # Run the prediction using TensorFlow Lite model
                    prediction = tflite_predict(interpreter, X_test.astype(np.float32))

                    # Decode the prediction result
                    decoded_prediction = decode_prediction(prediction, num_classes=10)
                    print('results: ', decoded_prediction, 'and image id: ', image['id'])

                    # Append the result with the image id
                    results.append({
                        'id': image['id'],
                        'number': decoded_prediction  # Predicted number as string
                    })

                except Exception as e:
                    print(e)
                    results.append({
                        'id': image['id'],
                        'error': str(e)
                    })

            return jsonify({'result': results}), 200
        except Exception as e:
            return jsonify({'error': 'An error occurred'}), 500

    return jsonify({'error': 'invalid request'}), 400



if __name__ == '__main__':
    app.run(debug=False)