import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize

def saveResults(filename, content):
    file_path  = 'results/train/' + filename;
    with open(file_path, 'w') as file:
        file.write(content)
        file.close()

def plotGraph(x_values, y_values, title, x_label, y_label, y_markings, path):
    plt.figure(figsize=(8, 5))
    for y_val, label in zip(y_values, y_markings):
        plt.plot(x_values, y_val, label=label, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # Save the plot as an image (e.g., PNG, JPG, PDF, etc.)
    plt.savefig(path + title + '.png')  # Change the file format as needed (e.g., .png, .jpg, .pdf)

class Log:
    def __init__(self):
        self.log_string = ""

    def log(self, message):
        print(message)
        self.log_string += message + "\n"

    def getLog(self):
        return self.log_string

class MetricsTracker:
    def __init__(self, auc = False):
        self.initialize()
        self.auc = auc
    
    def initialize(self):
        self.accuracy = 0
        self.loss = 0
        self.f1_macro = 0
        self.f1_weighted = 0
        self.class_total = defaultdict(int)
        self.class_correct = defaultdict(int)
        self.predicted_labels = []
        self.predicted_proba= []
        self.true_labels = []
        self.totals = 0
        self.corrects = 0
    
    def getF1Scores(self):
        return self.f1_macro, self.f1_weighted
    
    def getAUC(self):
        if not self.auc:
            return [], []
        auc_value_macro, auc_value_weighted = {}, {}
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predicted_proba)

        for i in range(3):
            auc = roc_auc_score(y_true[:, i], y_pred[:, i], average='macro')
            auc_value_macro[i] = auc
            auc = roc_auc_score(y_true[:, i], y_pred[:, i], average='weighted')
            auc_value_weighted[i] = auc
        return auc_value_macro, auc_value_weighted
    
    def getLoss(self):
        return self.loss, self.loss / self.totals
    
    def getCounts(self):
        return self.totals, self.corrects

    def getClassCounts(self):
        return self.class_total, self.class_correct
    
    def getAccurary(self):
        class_accuracy = {}
        for label, correct_count in self.class_correct.items():
            total_count = self.class_total[label]
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            class_accuracy[label] = accuracy * 100
        return self.accuracy, class_accuracy
    
    def addBatchResults(self, outputs, labels, loss):
        self.loss += loss.item()

        label_arg = torch.argmax(labels, dim = 1)
        predicted = torch.argmax(outputs, dim = 1)

        for x in outputs:
            one_hot = np.zeros(len(x), dtype=int)
            one_hot[torch.argmax(x)] = 1
            self.predicted_labels.append(one_hot)
            if self.auc:
                self.predicted_proba += [x.cpu().numpy()]

        for x in labels:
            cpu = x.cpu().numpy()
            self.true_labels += [cpu]
        
        for i in range(len(label_arg)):
            label = label_arg[i].item()
            pred = predicted[i].item()
            
            # Increment counts for the corresponding class
            self.class_total[label] += 1
            self.class_correct[label] += int(pred == label)

        self.totals += labels.size(0)
        self.corrects += (predicted == label_arg).sum().item()
        self.accuracy = (100 * self.corrects) / self.totals
        self.f1_macro = f1_score(self.true_labels, self.predicted_labels, average='macro')
        self.f1_weighted = f1_score(self.true_labels, self.predicted_labels, average='weighted')
    
    def plot_auc_curve(self, path):
        if not self.auc:
            return
        
        # Calculate AUC values
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predicted_proba)
        auc_values = []
        for i in range(3):  # Assuming 3 classes
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_values.append(auc)

        # Plot AUC curve
        plt.figure(figsize=(8, 6))
        plt.bar(range(3), auc_values, align='center')
        plt.xticks(range(3), ['Class 0', 'Class 1', 'Class 2'])  # Update labels as needed
        plt.xlabel('Classes')
        plt.ylabel('AUC')
        plt.title('AUC Curve for each class')
        plt.savefig(path + '_AUC.png')

    def plot_confusion_matrix(self, path):
        # Calculate confusion matrix
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predicted_labels)
        cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

        # Plot confusion matrix with numbers
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Class 0', 'Class 1', 'Class 2'],  # Update labels as needed
                    yticklabels=['Class 0', 'Class 1', 'Class 2'],  # Update labels as needed
                    )
        for i in range(len(cm)):
            for j in range(len(cm)):
                plt.text(j + 0.5, i + 0.5, str(cm[i][j]), ha='center', va='center', color='red')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(path + '_ConfusionMetrics.png')


    def plotROCCurve(self, path):
        if not self.auc:
            print("AUC metric is not enabled.")
            return

        y_true = label_binarize(self.true_labels, classes=[0, 1, 2])  # Update classes based on your task
        y_score = np.array(self.predicted_proba)

        n_classes = y_true.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for each class')
        plt.legend(loc="lower right")
        plt.savefig(path + '_ROCCurve.png')