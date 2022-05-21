Title:-
Classification of Legal court documents.

Introduction:-
Classifying legal unstructed court documents using NLP models, so that we can predict the outcomes of a court document. BERT, BOW(Bag-of-Words), and 
Fasttext were the different models used. Using the CLI we can predict the data, train our model and even obtain the results in csv format. 


Illustration:-
1. Our given code has two python files named main.py and predict_model.py.
2. We run the main.py file.
3. After running the code user can pass following commands to perform repective tasks:
	3.1. Command:-'--train' Action:- Train the models.
	3.2. Command:-'--predict' Action:- Predict the data.
	3.3. Command:-'--i' Action:- Set input path to the json file.
	3.4. Command:-'--t' Action:- Specify threshold between 0 and 1.
	3.5. Command:-'--p' Action:- Show number of probabilities.
	3.6. Command:-'--c' Action:- Create a csv file.
	3.7. Command:-'--bert' Action:- Create a BERT csv file.
	3.8. Command:-'--bow' Action:- Create a BOW csv file.
	3.9. Command:-'--ft' Action:- Create a Fasttext csv file.
4. We use pretrained models.
5. Data is read in the form of a csv file.
6. Data transformation is performed before the data is processed.

Libraries:-
json, argparse, numpy, csv, torch, pcikle, transformers.

Language:-
Python.