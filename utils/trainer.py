import torch
import numpy as np
from utils import utils
from utils.utils import Log, MetricsTracker
from tqdm import tqdm as tq
import torch.nn as nn
import time
from collections import defaultdict
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, model, name, learning_rate = 0.001, criterion =None, optimizer = None, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)
        self.name = name
        self.logger = Log()
        self.learning_rate = learning_rate
        self.list_of_training_accuracy = []
        self.list_of_validation_accuracy = []

        self.list_of_training_loss = []
        self.list_of_validation_loss = []

        self.list_of_training_f1 = []
        self.list_of_validation_f1 = []


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08) if optimizer is None else optimizer
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        


    def trainBatch(self, input_ids, attention_mask, labels, metrics):
        self.optimizer.zero_grad()
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        metrics.addBatchResults(outputs, labels, loss)

    def train(self, train_loader, val_loader=None, num_epochs=5, save = False):
        filename  = self.name +'_' +str(self.learning_rate) +'_'+ str(num_epochs)

        self.logger.log('-------------------------------------------')
        self.logger.log(f'Training model {self.name} . Learning Parameter {self.learning_rate}. num_of_epochs {num_epochs}')

        train_start_time = time.time()
        
        for epoch in range(num_epochs):

            metrics = MetricsTracker()
            self.logger.log(f'Epoch {epoch + 1}/{num_epochs}:')
            epoch_start_time = time.time()
            self.model.train()  # Set model to training mode

            with tq(train_loader,unit="batch") as td:
                for batch in td:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    self.trainBatch(input_ids, attention_mask, labels, metrics)

                    accuracy , _ = metrics.getAccurary()
                    _, weighted_f1 = metrics.getF1Scores()
                    td.set_description(desc = f"Accuracy={accuracy:.4f} f1:{weighted_f1:.4f}")
        

            train_accuracy, class_accuracy = metrics.getAccurary()
            f1_macro, f1_weighted = metrics.getF1Scores()
            train_loss , avg_train_loss = metrics.getLoss()
            train_total, train_correct = metrics.getCounts()

            self.list_of_training_accuracy.append(train_accuracy)
            self.list_of_training_loss.append(train_loss)
            self.list_of_training_f1.append(f1_weighted)

            self.logger.log(f'Training F1 score Weighted: {f1_weighted:.4f} Macro: {f1_macro:.4f}')
            self.logger.log(f'Training Avg Loss: {avg_train_loss:.4f}, Total Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Train Correct: {train_correct}, Train Total: {train_total}')

            # Validation step (if validation loader is provided)
            if val_loader is not None:
                self.validate(val_loader)

            epoch_end_time = time.time()
            
            epoch_time = epoch_end_time - epoch_start_time            
            self.logger.log(f"Epoch took: {epoch_time:.2f} seconds")

        train_end_time = time.time()
        # Calculate the elapsed time for the epoch
        train_time = train_end_time - train_start_time
        self.logger.log(f"Training took: {train_time:.2f} seconds")
        self.logger.log('-------------------------------------------')

        utils.saveResults(filename, self.logger.getLog())
        utils.plotGraph(np.arange(1, num_epochs + 1), [self.list_of_training_accuracy, self.list_of_validation_accuracy], 'Training and Validation Accuracy',
                        'Epoch', 'Accuracy', ['Training Accuracy', 'Validation Accuracy'], 'results/train/graphs/'+ filename)
        utils.plotGraph(np.arange(1, num_epochs + 1), [self.list_of_training_loss, self.list_of_validation_loss], 'Training and Validation Loss',
                        'Epoch', 'Loss', ['Training Loss', 'Validation Loss'], 'results/train/graphs/'+ filename)
        utils.plotGraph(np.arange(1, num_epochs + 1), [self.list_of_training_f1, self.list_of_validation_f1], 'Training and Validation F1',
                        'Epoch', 'F1', ['Training F1', 'Validation F1'], 'results/train/graphs/'+ filename)

        if save:
            torch.save(self.model.state_dict(), 'models/pytorch/' + filename + '.pth')

    def eval(self, test_loader, save_results = True):
        test_start_time = time.time()
        self.logger.log('-------------------------------------------')

        self.validate(test_loader, testing = True)

        test_end_time = time.time()
        self.logger.log('-------------------------------------------')


    def validate(self, val_loader, testing = False):
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        true_labels = []
        predicted_labels = []
        f1  = 0

        metrics = MetricsTracker(auc=testing)

        with torch.no_grad():
            with tq(val_loader,unit="batch") as td:
                for batch in td:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(labels, outputs)
                    metrics.addBatchResults(outputs, labels, loss)

                    val_accuracy, _ = metrics.getAccurary()
                    _, weighted_f1  = metrics.getF1Scores()

                    td.set_description(desc = f"Accuracy={val_accuracy:.4f} f1: {weighted_f1:.4f}")

        f1_macro, f1_weighted = metrics.getF1Scores()
        accuracy, class_acc = metrics.getAccurary()
        loss, avg_loss = metrics.getLoss()
        val_total, val_correct = metrics.getCounts()

        self.list_of_validation_accuracy.append(accuracy)
        self.list_of_validation_loss.append(loss)
        self.list_of_validation_f1.append(f1_weighted)

        self.logger.log(f'Validation Loss: {avg_loss:.4f}, Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}, val correct: {val_correct}, Val Total: {val_total}')
        self.logger.log(f'The F1 score : Weighted {f1_weighted:.4f} Macro {f1_macro:.4f}')

        if testing:
            path = 'results/test/graphs/' + self.name 
            auc_value_macro, auc_value_weighted = metrics.getAUC()
            metrics.plot_auc_curve(path)
            metrics.plot_confusion_matrix(path)
            metrics.plotROCCurve(path)
            self.logger.log(f'The AUC Score : Weighted {auc_value_weighted} Macro {auc_value_macro}')
        
        class_total, class_correct = metrics.getClassCounts()
        self.logger.log("Per-class accuracy:")
        for label, accuracy in class_acc.items():
            self.logger.log(f"Class {label}: {accuracy:.2f}% ({class_correct[label]}/{class_total[label]})")
