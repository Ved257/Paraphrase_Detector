from googletrans import Translator
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from http.client import HTTPException

# Create an instance of the Translator class
translator = Translator()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Text to be translated
@app.route('/translate_api', methods=['POST'])
def translate():
    input = request.get_json(force=True)
    print(list(input.values()))
    inputs = (list(input.values()))
    text_to_translate = inputs[0]

    # Detect the language of the text
    detected_language = translator.detect(text_to_translate)
    print(f"Detected Language: {detected_language.lang}")

    # Translate the text to a specified language
    target_language = 'en'  # 'es' is the language code for Spanish
    translated_text = translator.translate(text_to_translate, dest=target_language)

    # Print the translated text and the source and destination languages
    print(f"Original Text: {text_to_translate}")
    print(f"Translated Text: {translated_text.text}")
    print(f"Source Language: {translated_text.src}")
    print(f"Destination Language: {translated_text.dest}")
    return jsonify(translated_text.text)

@app.errorhandler(HTTPException)
def handle_exception(e):
    return jsonify({"message": e.description}), e.code


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)
