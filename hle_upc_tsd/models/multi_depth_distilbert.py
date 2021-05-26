from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule
from transformers import DistilBertModel, AdamW

from ..utils.custom_losses import CEL_label_smoothing
from ..utils.metrics import f1_score
from ..utils.preds2spans import preds2spans


class MultiDepthDistilBertModel(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters('args')

        # Hyperparameters
        self.label_smoothing = args.label_smoothing
        self.lr = args.lr
        self.concat_last_n = args.concat_last_n

        # Model
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                    output_attentions=False,
                                                    output_hidden_states=True,
                                                    dropout=args.dropout)
        self.drop = torch.nn.Dropout(args.dropout)
        self.linear = torch.nn.Linear(self.concat_last_n * self.bert.config.dim, 2)

        # Predictions
        self.predictions = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--dropout', type=float, default=0.25)
        parser.add_argument('--label_smoothing', type=float, default=0.1)
        parser.add_argument('--concat_last_n', type=int, default=4)

        return parser

    def forward(self, input_ids, attention_mask):
        batch_size, seq_size = input_ids.shape
        # Use bert model
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Concatenate last n layers
        hidden = torch.cat(output.hidden_states[-self.concat_last_n:], dim=-1)
        # Apply dropout and classification layer to every element in the sequence
        hidden = hidden.view(batch_size * seq_size, -1)
        hidden = self.drop(hidden)
        logits = self.linear(hidden)
        logits = logits.view(batch_size, seq_size, -1)

        return logits

    def training_step(self, batch, batch_idx):
        token_ids, att_masks, label_ids, offsets, original_spans, special_masks = batch
        special_masks = special_masks.logical_not()
        batch_size, seq_size = token_ids.shape

        logits = self(token_ids, att_masks)

        loss = CEL_label_smoothing(logits.reshape(batch_size * seq_size, -1), label_ids.view(-1),
                                   smoothing=self.label_smoothing, mask=special_masks.view(-1))

        preds = torch.argmax(logits, -1)

        predicted_spans = preds2spans(preds, special_masks, offsets)
        f1 = f1_score(predicted_spans, original_spans)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.log('train_f1', f1, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        token_ids, att_masks, label_ids, offsets, original_spans, special_masks = batch
        special_masks = special_masks.logical_not()
        batch_size, seq_size = token_ids.shape

        logits = self(token_ids, att_masks)

        loss = CEL_label_smoothing(logits.reshape(batch_size * seq_size, -1), label_ids.view(-1),
                                   smoothing=self.label_smoothing, mask=special_masks.view(-1))

        preds = torch.argmax(logits, -1)

        predicted_spans = preds2spans(preds, special_masks, offsets)
        f1 = f1_score(predicted_spans, original_spans)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        token_ids, att_masks, label_ids, offsets, original_spans, special_masks = batch
        special_masks = special_masks.logical_not()

        logits = self(token_ids, att_masks)

        preds = torch.argmax(logits, -1)

        predicted_spans = preds2spans(preds, special_masks, offsets)

        return {'predicted_spans': predicted_spans,
                'original_spans': original_spans}

    def test_epoch_end(self, outputs):
        predictec_spans = []
        original_spans = torch.LongTensor().to(self.device)
        # Accumulate data from all batches
        for batch in outputs:
            predictec_spans.extend(batch['predicted_spans'])
            original_spans = torch.cat([original_spans, batch['original_spans']], dim=0)

        # Store information
        self.predictions = {'predicted_spans': predictec_spans,
                            'original_spans': original_spans}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
