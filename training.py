
import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
from torch.utils.data import DataLoader
import time
import warnings
from collections import defaultdict
import traceback
from tqdm import tqdm as tq
from models.transformers.classifier import FNN1LayerClassifier, FNN2LayerClassifier
from models.transformers.transformers import BaseBertClassifier, BaseBertLayerwiseClassifier
from utils import preprocessor, config, utils
from utils.preprocessor import YelpDataset
from utils.trainer import Trainer
import argparse

MAX_LENGTH = 256
BATCH_SIZE = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def buildAndTrain(name, model, train_loader, val_loader, device, lr, num_epochs, save):
   
    try:
        trainer = Trainer(model, name=name, learning_rate = lr, device = device)
        trainer.train(train_loader, val_loader, num_epochs = num_epochs, save = save)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        traceback.print_exc()
        print(error_message)
        file_path = "error_log.txt"

        with open(file_path, 'a') as file:
            file.write(error_message + "\n")
            file.close()
def prepData():
    # Set the environment variable for the cache directory
    os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "/scratch/user/sathvikkote/ml/.cache"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    file_path = 'yelp_review_train.csv'
    yelp_data = pd.read_csv(file_path)

    train_texts, val_texts, train_labels, val_labels = preprocessor.preprocessData(yelp_data, balance= True)

    # Create instances of custom datasets for training and validation
    train_dataset = YelpDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = YelpDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader


def classifierDesign(saveModel = False):
    print(f'Save Model : {saveModel}')
    print('Device: ',torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    train_loader, val_loader = prepData()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
    name = 'baseBertWith2LayerClassfier'
    baseBertWith2layerClassifier = BaseBertLayerwiseClassifier(pretrained_bert, FNN2LayerClassifier(3))
    buildAndTrain( name, baseBertWith2layerClassifier, train_loader, val_loader, device, 2e-05, 8, saveModel)

    pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
    name = 'baseBertWith1LayerClassfier'
    baseBertWith1LayerClassfier = BaseBertLayerwiseClassifier(pretrained_bert, FNN1LayerClassifier(3))
    buildAndTrain(name, baseBertWith1LayerClassfier, train_loader, val_loader, device, 2e-05, 8, saveModel)

def parameterTuning(saveModel = False):
    print(f'Save Model : {saveModel}')
    print('Device: ',torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for length in [512, 256]:
        for batch in [16 , 8]:
            BATCH_SIZE = batch
            MAX_LENGTH = length

            train_loader, val_loader = prepData()
            pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
            name = 'baseBertWith2LayerClassfier_' + str(length) + '_' + str(batch)
            baseBertWith2layerClassifier = BaseBertLayerwiseClassifier(pretrained_bert, FNN2LayerClassifier(3))
            buildAndTrain( name, baseBertWith2layerClassifier, train_loader, val_loader, device, 2e-05, 8, saveModel)

def layerSelection(saveModel = False):
    print(f'Save Model : {saveModel}')
    print('Device: ',torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    train_loader, val_loader = prepData()

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for layer in range(0,13):
        pretrained_bert = BertModel.from_pretrained('bert-base-uncased',  output_hidden_states=True)
        bertwith2fnn = BaseBertLayerwiseClassifier(pretrained_bert,  FNN2LayerClassifier(3), layer)
        name = 'bert_with_2_fnn_' + str(layer) + '_layer'
        buildAndTrain(name, bertwith2fnn, train_loader, val_loader, device, 2e-05, 3, saveModel)

def freezingUnfreezingTraining(saveModel = False):
    print('Save Model ',saveModel)
    print('Device: ',torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    train_loader, val_loader = prepData()

    pretrained_bert = BertModel.from_pretrained('bert-base-uncased',  output_hidden_states=True)
    name = 'baseBertWith2LayerClassfierUsing12Layer'
    baseBertWith2LayerClassfierUsing12Layer = BaseBertLayerwiseClassifier(pretrained_bert, FNN2LayerClassifier(3), layer = 12)
    buildAndTrain(name, baseBertWith2LayerClassfierUsing12Layer, train_loader, val_loader, device, 2e-05, 4, saveModel)

    pretrained_bert = BertModel.from_pretrained('bert-base-uncased',  output_hidden_states=True)
    name = 'baseBertWith2LayerClassfierUsing12LayerUnFreeze'
    baseBertWith2LayerClassfierUsing12LayerUnFreeze = BaseBertLayerwiseClassifier(pretrained_bert,  FNN2LayerClassifier(3), 12, freeze = False)
    baseBertWith2LayerClassfierUsing12LayerUnFreeze.load_state_dict(torch.load('models/pytorch/baseBertWith2LayerClassfierUsing12Layer_2e-05_4.pth'))
    buildAndTrain(name, baseBertWith2LayerClassfierUsing12LayerUnFreeze, train_loader, val_loader, device, 2e-05, 4, saveModel)

def evaluate(path, classifier_layers, name, layer):
    if path is None:
        return
    if classifier_layers is None:
        return
    if name is None:
        return
    print('Started Evaluating')
    print(f'Path: {path} Name: {name} Classifier : {classifier_layers} layer : {layer}')

    if int(classifier_layers) == 1:
        classifier = FNN1LayerClassifier(3)
    else:
        classifier =  FNN2LayerClassifier(3)

    pretrained_bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    if layer:
        model = BaseBertLayerwiseClassifier(pretrained_bert, classifier, layer = int(layer[0]))
    else:
        model = BaseBertClassifier(pretrained_bert, classifier)
        
    model.load_state_dict(torch.load(path))

    file_path = 'yelp_review_test.csv'
    yelp_data = pd.read_csv(file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_texts, val_texts, train_labels, val_labels = preprocessor.preprocessData(yelp_data, test_size = 0.0)

    # Create instances of custom datasets for training and validation
    test_dataset = YelpDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    trainer = Trainer(model, name = name,device = device)
    trainer.eval(test_loader)
    print('Finished Evaluation')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Training and evaluation script')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', nargs=1, metavar=('step'), help='Train the model')
    group.add_argument('--eval', nargs='+', metavar=('path', 'model', 'name', 'layer'), help='Evaluate the model with the provided path and model value (1 or 2)')
    parser.add_argument('--save', action='store_true', help='Save the model')
    args = parser.parse_args()
    
    # Training
    if args.train:
        step = args.train[0]
        if int(step) == 1:
            # Classifier Design
            classifierDesign(args.save)
        if int(step) == 2:
            # Parameter Tunning
            parameterTuning(args.save)
        if int(step) == 3:
            # Layer wise training
            layerSelection(args.save)
        if int(step) == 4:
            # Freeze and Unfreeze
            freezingUnfreezingTraining(args.save)

    # Evaluation
    if args.eval:
        path, model_value, name, *layer = args.eval
        if layer:
            evaluate(path, model_value, name, layer)
        else:
            evaluate(path, model_value, name, None)