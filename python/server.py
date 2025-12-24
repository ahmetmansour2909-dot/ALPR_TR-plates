from flask import Flask, request, jsonify
from alpr_pipeline import process_plate_image

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def ocr_api():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]

    try:
        result = process_plate_image(image_file)
        return jsonify({"plate": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
