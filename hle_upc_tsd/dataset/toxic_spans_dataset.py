import os
from ast import literal_eval
from csv import DictReader

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import DistilBertTokenizerFast

from ..utils.fix_spans import my_fix_spans


class ToxicDataset(Dataset):
    def __init__(self, data_path, split):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        self.original_sentences, self.original_spans, self.fixed_spans = \
            self.get_sentences_from_data_split(data_path, split)

        self.token_ids, self.offsets, self.att_masks, self.special_masks, self.labels_ids = \
            self.preprocess_and_tokenize(self.original_sentences, self.fixed_spans)

    def __len__(self):
        return len(self.original_sentences)

    def __getitem__(self, index):
        token_ids = self.token_ids[index]
        offsets = self.offsets[index]
        att_masks = self.att_masks[index]
        special_masks = self.special_masks[index]
        label_ids = self.labels_ids[index]
        original_spans = self.original_spans[index]
        # Add padding to original_spans, which is the only one that is not padded yet.
        # All span sets are shorter than 1024
        original_spans.extend([-1] * (1024 - len(original_spans)))

        # To Tensor
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        offsets = torch.tensor(offsets, dtype=torch.long)
        att_masks = torch.tensor(att_masks, dtype=torch.long)
        special_masks = torch.tensor(special_masks, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        original_spans = torch.tensor(original_spans, dtype=torch.long)

        return token_ids, att_masks, label_ids, offsets, original_spans, special_masks

    @staticmethod
    def get_sentences_from_data_split(data_path, split):
        sentences = []
        original_spans = []
        fixed_spans = []
        with open(os.path.join(data_path, split + '.csv'), encoding='utf-8') as csv_file:
            reader = DictReader(csv_file)
            for row in reader:
                if split == 'tsd_test':
                    span = fixed_span = []
                else:
                    span = literal_eval(row['spans'])
                    fixed_span = my_fix_spans(span, row['text'])
                sentences.append(row['text'])
                original_spans.append(span)
                fixed_spans.append(fixed_span)

        return sentences, original_spans, fixed_spans

    def preprocess_and_tokenize(self, sentences, spans):
        all_token_ids = []
        all_offsets = []
        all_att_masks = []
        all_special_masks = []
        all_label_ids = []

        for sentence, span in zip(sentences, spans):
            # Pad to 512. All sentences in the dataset have a lower number of tokens.
            tokenized = self.tokenizer(sentence, padding='max_length', max_length=512, return_attention_mask=True,
                                       return_special_tokens_mask=True,
                                       return_offsets_mapping=True, return_token_type_ids=False)

            all_token_ids.append(tokenized['input_ids'])
            all_offsets.append(tokenized['offset_mapping'])
            all_att_masks.append(tokenized['attention_mask'])
            all_special_masks.append(tokenized['special_tokens_mask'])
            all_label_ids.append([self.off2tox(offset, span) for offset in tokenized['offset_mapping']])

        return all_token_ids, all_offsets, all_att_masks, all_special_masks, all_label_ids

    @staticmethod
    def off2tox(offsets, spans):
        # Padded items
        if offsets == (0, 0):
            return 0
        toxicity = offsets[0] in spans
        return int(toxicity)


class ToxicDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root, batch_size, num_workers):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_set = ToxicDataset(self.dataset_root, split='tsd_train')
            self.val_set = ToxicDataset(self.dataset_root, split='tsd_trial')
        elif stage == 'val':
            self.test_set = ToxicDataset(self.dataset_root, split='tsd_trial')
        elif stage == 'test':
            self.test_set = ToxicDataset(self.dataset_root, split='tsd_test_gt')
        elif stage == 'all':
            self.test_set = ToxicDataset(self.dataset_root, split='tsd_all_test')

    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.num_workers, shuffle=True, batch_size=self.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, num_workers=self.num_workers, shuffle=False, batch_size=self.batch_size,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, num_workers=self.num_workers, shuffle=False, batch_size=self.batch_size,
                          pin_memory=True)

    def prepare_data(self, *args, **kwargs):
        pass
