from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)

loaded_nlp = spacy.load("items-pdf-model")


# loaded_nlp = spacy.load("items-pdf-model")


def extract_info(text):
    doc = loaded_nlp(text)
    extracted_info = {
        "NAME": [],
        "SIZE": [],
        "PRICE": [],
        "ITEM_ID": []
    }

    for ent in doc.ents:
        if ent.label_ in extracted_info:
            extracted_info[ent.label_].append(ent.text)

    return extracted_info


def process_multiple_texts(texts):
    results = []
    for text in texts:
        result = extract_info(text)
        if any(result.values()):
            results.append(result)
    return results


@app.route('/items-pdf/extract', methods=['POST'])
def extract():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        if not isinstance(texts, list):
            return jsonify({"error": "Invalid input, 'texts' should be a list of strings."}), 400
        results = process_multiple_texts(texts)
        return jsonify(results)
    except Exception as e:
        print(f'Exception Occured {e}')


if __name__ == '__main__':
    app.run(port=5555)
