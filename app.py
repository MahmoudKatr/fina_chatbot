from flask import Flask, request, jsonify
from chat import get_response
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.get("/predict")
@cross_origin()
def predict():
    text = request.args.get("message")

    if text:
        response = get_response(text)
        message = {"answer": response}
        return jsonify(message)
    else:
        return jsonify({"error": "No message provided"}), 400

if __name__ == "__main__":
    app.run(debug=True)

## new ##
# from flask import Flask, request, jsonify
# from chat import get_response
# from flask_cors import CORS, cross_origin

# app = Flask(__name__)

# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# @app.get("/predict")
# @cross_origin()
# def predict():
#     text = request.args.get("message")

#     if text:
#         response = get_response(text)
#         message = {"answer": response}
#         return jsonify(message)
#     else:
#         return jsonify({"error": "No message provided"}), 400

# if __name__ == "__main__":
#     app.run(debug=True)