import torch
import torch.nn as nn

class BaseBertClassifier(nn.Module):
    def __init__(self, bert_model, linear_fnn, freeze = True):
        super(BaseBertClassifier, self).__init__()
        self.bert = bert_model
        self.fnn = linear_fnn

        # Freeze BERT layers
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Unfreeze the feedforward neural network layers for training
        for param in self.fnn.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use pooler_output from BERT
        x = self.fnn(pooled_output)
        return x

class BaseBertLayerwiseClassifier(nn.Module):
    def __init__(self, bert_model, linear_fnn, layer = None, freeze = True):
        super(BaseBertLayerwiseClassifier, self).__init__()
        self.bert = bert_model
        self.fnn = linear_fnn
        self.layer = layer

        # Freeze BERT layers
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Unfreeze the feedforward neural network layers for training
        for param in self.fnn.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.layer is None:
            pooled_output = outputs.pooler_output  # Use pooler_output from BERT
            x = self.fnn(pooled_output)
            return x
        else:
            hidden_states = outputs[2]  # Get all hidden states (layers)

            # Select the output of the 12th layer (index 12, as index 0 represents the input embeddings)
            bert_output_ith_layer = hidden_states[self.layer]  # Output of the 12th layer

            # Extract the [CLS] token embedding (index 0) for classification
            cls_token_embeddings = bert_output_ith_layer[:, 0, :]  # Select the [CLS] token embedding

            # Pass the [CLS] token embedding through the feedforward neural network
            x = self.fnn(cls_token_embeddings)
            return x