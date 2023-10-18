from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from flask import Flask

app = Flask(__name__)

# Load the pre-trained Xception model outside of the route function
xception_model = Xception(weights='imagenet', include_top=True)

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"})

    video_file = request.files['video']
    # You can specify where to save the video file, or process it directly in memory.

    # Example: Save the video file to disk
    video_file.save('uploaded_video.mp4')
#    pdb.set_trace()  # Set a breakpoint

    # Example: Process the video using OpenCV and Xception model
    cap = cv2.VideoCapture('uploaded_video.mp4')
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to the input size of the Xception model
        if frame is not None:
            frame = cv2.resize(frame, (299, 299))

            # Preprocess the frame
            x = tf.image.convert_image_dtype(frame, tf.float32)
            x = tf.image.resize(x, (299, 299))
            x = preprocess_input(x)
            x = tf.expand_dims(x, axis=0)

            # Make predictions using the Xception model
            predictions = xception_model.predict(x)

            # Decode the predictions (optional)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            # Extract labels and confidences
            frame_results = [{"label": label, "confidence": float(confidence)} for (_, label, confidence) in decoded_predictions]

            results.append(frame_results)

    cap.release()
#    cv2.destroyAllWindows()

    return jsonify({"Video analysis completed": results})

if __name__ == '__main__':
    app.run(debug=True)

