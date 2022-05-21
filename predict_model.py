import fasttext
import json
from json.encoder import JSONEncoder
import numpy as np
import csv
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
import warnings
from predict_preprocess import *

logging.set_verbosity_error()
warnings.filterwarnings('ignore')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_predict_data(path):
    file = open(path,)
    data = json.load(file)
    newdata = []
    for d in data['elements']:
        newdata.append(preprocess_predict(d['Text']))
    return newdata


def ft_predict(path, create_csv, threshold):

    data = get_predict_data(path)

    # Fasttext Json file
    fl = open('fasttext_output.json', 'w', encoding='utf-8')
    csvdata = []

    # Call the model
    fst_mdl = fasttext.load_model(
        r'../script/predict_models/fastText/model_fasttext.bin')

    # Predict for each sentence in the data
    for d in data:
        word_list = d.split()

        # Clean string and keep only sentences with words greater than 1
        if(len(word_list) > 1):

            output = fst_mdl.predict(d, k=1)

            if(output[1] >= threshold):
                if(create_csv):
                    csvdata.append([d, output[0], output[1]])
                x = {
                    "text": d, "label": output[0], "probability": output[1]
                }
                json.dump(x, fl, cls=NumpyArrayEncoder)
                fl.write("\n")

        # Create the csv if true
        if(create_csv):
            create_csv_def(csvdata, 'fasttext_output.csv')

    return True


def bert_predict(path, create_csv, threshold):

    data = get_predict_data(path)

    # Bert Json file
    fl = open('bert_output.json', 'w', encoding='utf-8')
    csvdata = []

    # Call the model and tokenizer
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_names = ['dec_name', 'counsel', 'court', 'facts', 'judge', 'outcome']
    loaded_model = AutoModelForSequenceClassification.from_pretrained(
        "predict_models/legal_bert")

    # Predict for each sentence in the data
    for d in data:

        word_list = d.split()

        # Clean string and keep only sentences with words greater than 1
        if(len(word_list) > 1):

            inputs = tokenizer(d, padding="max_length",
                               truncation=True, return_tensors="pt")

            outputs = loaded_model(**inputs)
            probs = torch.nn.functional.softmax(outputs['logits'], dim=1)
            probs = probs.detach().numpy().flatten()
            sort_probs = np.sort(probs)[::-1]
            ordered_labels = []
            for prob in sort_probs:
                x = np.array(np.where(probs == prob)).flatten()
                ordered_labels.append(label_names[x[0]])

            if(sort_probs[0] >= threshold):

                if(create_csv):
                    csvdata.append([d, ordered_labels[0], sort_probs[0]])

                x = {
                    "text": d, "label": ordered_labels[0], "probability": str(sort_probs[0])
                }
                json.dump(x, fl, cls=NumpyArrayEncoder)
                fl.write("\n")

        # Create the csv if true
        if(create_csv):
            create_csv_def(csvdata, 'bert_output.csv')

    return True


def bow_predict(path, create_csv, threshold):

    data = get_predict_data(path)

    # Bow Json file
    fl = open('bow_output.json', 'w', encoding='utf-8')
    csvdata = []

    # Call the model
    mdl = pickle.load(
        open('../script/predict_models/bag_of_words/bow.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(
        open('../script/predict_models/bag_of_words/vectorize.pkl', 'rb'))

    # Predict for each sentence in the data
    for d in data:
        word_list = d.split()

        # Clean string and keep only sentences with words greater than 1
        if(len(word_list) > 1):

            X_test = [d]
            X_vector = tfidf_vectorizer.transform(
                X_test)  # converting X_test to vector
            y_predict = mdl.predict(X_vector)
            arr = mdl.predict_proba(X_vector)[:, :]
            y_prob = arr[0][np.argmax(arr)]

            if(y_prob >= threshold):
                if(create_csv):
                    csvdata.append([d, y_predict[0], y_prob])

                x = {
                    "text": d, "label": y_predict[0], "probability": y_prob
                }
                json.dump(x, fl, cls=NumpyArrayEncoder)
                fl.write("\n")

        # Create the csv if true
        if(create_csv):
            create_csv_def(csvdata, 'bow_output.csv')
    return True


def create_csv_def(csv_data, filename):
    header = ["text", "label", "probability"]
    with open(filename, 'w', encoding='utf-8', newline='') as flcsv:
        writer = csv.writer(flcsv)
        writer.writerow(header)
        writer.writerows(csv_data)
