import json
import argparse
from os import system
from predict_preprocess import *
from predict_model import *
from train_model import *


parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true',
                    dest='train_model',
                    help='Option to train the models')

parser.add_argument('--predict', action='store_true',
                    dest='predict_model',
                    help='Option to predict the data')

parser.add_argument('-i', action='store', dest='input_path',
                    help='Set input path to the json file')

parser.add_argument('-t', action='store', dest='threshold', type=float,
                    default=0.0, help='Specify threshold between 0 and 1')

parser.add_argument('-p', action='store', default=1,
                    dest='num_prob', type=int,
                    help='Show number of probabilites')

parser.add_argument('-c', action='store_true',
                    dest='create_csv',
                    help='Creates a csv file')

parser.add_argument('--bert', action='store_true',
                    dest='bert',
                    help='Creates a csv file')

parser.add_argument('--bow', action='store_true',
                    dest='bow',
                    help='Creates a csv file')

parser.add_argument('--ft', action='store_true',
                    dest='ft',
                    help='Creates a csv file')


results = parser.parse_args()

if (results.train_model == False) and (results.predict_model == False):
    system.exit('Please specify --train or --predict argument')


if(results.predict_model):
    if(results.ft):
        print("Model : FastText \nCreate CSV file: ",
              results.create_csv, " \nThreshold : ", results.threshold)

        success = ft_predict(results.input_path,
                             results.create_csv, results.threshold)

        if(success):
            print("FastText output successfull!!")
        else:
            print("FAILED!!")

    if(results.bert):
        print("Model : BERT \nCreate CSV file: ",
              results.create_csv, " \nThreshold : ", results.threshold)

        success = bert_predict(
            results.input_path, results.create_csv, results.threshold)

        if(success):
            print("BERT output successfull!!")
        else:
            print("FAILED!!")

    if(results.bow):
        print("Model : BagOfWrods \nCreate CSV file: ",
              results.create_csv, " \nThreshold : ", results.threshold)

        success = bow_predict(results.input_path,
                              results.create_csv, results.threshold)

        if(success):
            print("BagOfWords output successfull!!")
        else:
            print("FAILED!!")


if(results.train_model):
    if(results.ft):
        print("Model : FastText")

        success = ft_train(results.input_path)

        if(success):
            print("FastText output successfull!!")
        else:
            print("FAILED!!")

    if(results.bert):
        print("Model : BERT")

        success = bert_train(results.input_path)

        if(success):
            print("BERT output successfull!!")
        else:
            print("FAILED!!")

    if(results.bow):
        print("Model : BagOfWrods ")

        success = bow_train(results.input_path)

        if(success):
            print("BagOfWords output successfull!!")
        else:
            print("FAILED!!")
