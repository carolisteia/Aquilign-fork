# -*- coding: utf-8 -*-
"""
tok_trainer_functions.py
------------------------
Helper functions and dataset class used for training a segmentation
model with HuggingFace Transformers.

This module contains:
    - Utilities to align word-level segmentation labels with
      subword tokens after BERT tokenization
    - A Dataset class (SentenceBoundaryDataset) for PyTorch
    - Functions to compute evaluation metrics (accuracy, precision, recall, f1)

Noise injection:
    The Dataset optionally applies artificial noise to input texts
    (via aquilign.preproc.noise.apply_noise) in order to simulate
    less regularized inputs such as HTR outputs.
"""

import random
import re
import torch
import evaluate
import numpy as np
from torch.utils.data import Dataset
from aquilign.preproc.noise import apply_noise

# -------------------------------------------------------------------
# Language-specific noise configuration
# -------------------------------------------------------------------
# Each language can have its own noise settings.
# - apply_noise (bool): whether to apply noise
# - noise_prob (float): probability that an example is noised
# - noise_level (str): 'light' | 'medium' | 'heavy'
#
# Example strategy:
#   - French (fr): mostly clean editions → more noise
#   - Portuguese (pt): already semi-diplomatic → less noise
#   - Latin (la): moderately clean → medium noise
#   - English (en): balanced → light noise
#   - Catalan (ca): lighter noise (texts often already less normalized)
#   - Italian (it): medium noise
#   - Castilian (es): medium noise
# -------------------------------------------------------------------
LANG_NOISE_CONFIG = {
    "fr": {"apply_noise": True, "noise_prob": 0.5, "noise_level": "medium"},
    "pt": {"apply_noise": True, "noise_prob": 0.1, "noise_level": "light"},
    "la": {"apply_noise": True, "noise_prob": 0.3, "noise_level": "medium"},
    "en": {"apply_noise": True, "noise_prob": 0.2, "noise_level": "light"},
    "ca": {"apply_noise": True, "noise_prob": 0.2, "noise_level": "light"},
    "it": {"apply_noise": True, "noise_prob": 0.3, "noise_level": "medium"},
    "es": {"apply_noise": True, "noise_prob": 0.3, "noise_level": "medium"},
}

# Global dictionary to count how many debug prints have been shown per language
DEBUG_NOISE_COUNTER = {}
DEBUG_NOISE_LIMIT = 5   # max prints per language

# -------------------------------------------------------------------
# Utility: index correspondence between raw words and BERT tokens
# -------------------------------------------------------------------
def get_index_correspondence(sent, tokenizer):
    """
    For a list of words, compute how they expand into BERT subword tokens.

    Args:
        sent (list of str): input words
        tokenizer: HuggingFace tokenizer

    Returns:
        correspondence (list of tuples):
            each tuple (raw_end, expand_end) gives index alignment
            between original word sequence and expanded subwords
    """
    correspondence = [(0, 0)]
    for word in sent:
        (raw_end, expand_end) = correspondence[-1]
        tokenized_word = tokenizer.tokenize(word)
        correspondence.append((raw_end + 1, expand_end + len(tokenized_word)))
    return correspondence


# -------------------------------------------------------------------
# Utility: align gold segmentation labels with subword tokens
# -------------------------------------------------------------------
def align_labels(corresp, orig_labels, text):
    """
    Expand word-level segmentation labels to token-level labels
    after BERT tokenization.

    Args:
        corresp (list): index correspondence from get_index_correspondence()
        orig_labels (list of int): gold labels at word-level
            0 = no boundary
            1 = boundary
        text (str): original text (for debugging if error occurs)

    Returns:
        new_labels (list of int):
            labels expanded to match BERT tokens
            special tokens (CLS/SEP) are assigned label = 2
    """
    new_labels = [0 for _ in range(corresp[-1][1])]
    for index, label in enumerate(orig_labels):
        if label == 1:
            try:
                if len(new_labels) == corresp[index][1]:
                    # mark the last token of the word as boundary
                    new_labels[(corresp[index][1]) - 1] = 1
            except IndexError:
                print(new_labels)
                print("Error.")
                exit(0)
            else:
                try:
                    new_labels[(corresp[index][1])] = 1
                except IndexError:
                    print(f"Error with example:\n {text}.\nExiting.")
        else:
            pass

    # add special tokens (CLS/SEP) with value 2
    new_labels.insert(0, 2)
    new_labels.append(2)
    return new_labels


# -------------------------------------------------------------------
# Utility: compute max tokenized sequence length
# -------------------------------------------------------------------
def get_token_max_length(train_texts, tokenizer):
    """
    Compute the maximum sequence length after tokenization,
    used to set padding/truncation length.

    Args:
        train_texts (list of str): training texts
        tokenizer: HuggingFace tokenizer

    Returns:
        max_length (int): maximum tokenized length
    """
    lengths_list = []
    for text in train_texts:
        tok_text = tokenizer(text, return_tensors='pt')
        tensor_length = (tok_text['input_ids'].squeeze())
        length = tensor_length.shape[0]
        lengths_list.append(length)
    return max(lengths_list)



# -------------------------------------------------------------------
# Dataset class: SentenceBoundaryDataset
# -------------------------------------------------------------------
class SentenceBoundaryDataset(Dataset):
    """
    PyTorch Dataset for segmentation.

    - Expects as input a list of dicts (already tokenized & aligned):
        each item has keys 'input_ids', 'attention_mask', 'labels'
    - Optionally applies noise (before tokenization if you want raw text pipeline)
    - Currently, noise is only logged (since data is already tokenized)

    Args:
        texts_and_labels (list of dicts): from utils.convertToSubWordsSentencesAndLabels()
        tokenizer: HuggingFace tokenizer (kept for compatibility)
        lang (str): language code (for language-specific noise configs)
        debug_noise (bool): if True, prints noisy examples
    """

    def __init__(self, texts_and_labels, tokenizer, lang="fr", debug_noise=False):
        self.data = texts_and_labels
        self.tokenizer = tokenizer
        self.lang = lang
        self.debug_noise = debug_noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Already tokenized dict with input_ids, attention_mask, labels
        item = self.data[idx]

        # ⚠️ At this stage text is already tokenized, so we cannot apply
        # noise at character level anymore unless we move it earlier
        # (before convertToSubWordsSentencesAndLabels).
        #
        # For now: just pass through
        return {key: val.squeeze(0) if hasattr(val, "squeeze") else val 
                for key, val in item.items()}


# class SentenceBoundaryDataset(Dataset):
#     """
#     PyTorch Dataset for segmentation.

#     - Loads (text, labels) pairs
#     - Optionally applies noise to simulate HTR-like input
#       (language-specific probabilities and levels)
#     - Tokenizes text with BERT tokenizer
#     - Aligns segmentation labels to subword tokens

#     Args:
#         texts_and_labels (list): list of (text, labels) tuples
#             - text (str): input string (a sentence or passage)
#             - labels (list of int): gold segmentation labels
#                 - 0 = no boundary
#                 - 1 = boundary
#         tokenizer: HuggingFace tokenizer
#         lang (str): language code (e.g. "fr", "pt", "la", "es")
#         debug_noise (bool): if True, print a few noisy samples

#     Notes
#     -----
#     • Noise is applied at the *example level* (not the batch).  
#     • It happens *before tokenization*, so the tokenizer sees noisy text.  
#     • This simulates realistic HTR-like inputs (char confusions, deletions, missing punctuation).  
#     • Each language has its own configuration in LANG_NOISE_CONFIG.  
#     """

#     def __init__(self, texts_and_labels, tokenizer, lang="fr", debug_noise=False):
#         self.data = texts_and_labels
#         self.tokenizer = tokenizer
#         self.lang = lang
#         self.debug_noise = debug_noise

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text, labels = self.data[idx]

#         # -----------------------------------------------------------
#         # Language-specific noise injection
#         # - Controlled by LANG_NOISE_CONFIG
#         # - Each language can have different noise_prob and noise_level
#         # - Uses apply_noise() from aquilign/preproc/noise.py
#         # -----------------------------------------------------------
#         cfg = LANG_NOISE_CONFIG.get(self.lang, {"apply_noise": False})

#         if cfg.get("apply_noise") and random.random() < cfg.get("noise_prob", 0.0):
#             text, labels = apply_noise(
#                 text, labels, noise_level=cfg.get("noise_level", "medium")
#             )

#             # Debug printing: show only first 5 noisy samples per language
#             if self.debug_noise:
#                 count = DEBUG_NOISE_COUNTER.get(self.lang, 0)
#                 if count < DEBUG_NOISE_LIMIT:
#                     print(f"[NOISE DEBUG: {self.lang.upper()} | "
#                           f"prob={cfg.get('noise_prob')} | level={cfg.get('noise_level')}] "
#                           f"{text}")
#                     DEBUG_NOISE_COUNTER[self.lang] = count + 1

#         # -----------------------------------------------------------
#         # Tokenization
#         # - Converts text into subword tokens compatible with BERT
#         # - truncation/padding ensures fixed length
#         # - word_ids used to align segmentation labels
#         # -----------------------------------------------------------
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             max_length=128,
#             return_tensors="pt",
#             is_split_into_words=False
#         )

#         # Align labels with tokens
#         word_ids = encoding.word_ids(batch_index=0)
#         token_labels = []
#         for word_id in word_ids:
#             if word_id is None:
#                 token_labels.append(-100)   # ignore special tokens
#             else:
#                 token_labels.append(labels[word_id])
#         encoding["labels"] = token_labels

#          # -----------------------------------------------------------
#         # HuggingFace Trainer expects dict of tensors
#         # - .squeeze(0) removes the extra batch dimension
#         # -----------------------------------------------------------

#         # Squeeze to remove extra batch dimension
#         return {key: val.squeeze(0) for key, val in encoding.items()}


   

# -------------------------------------------------------------------
# Metrics computation
# -------------------------------------------------------------------
# def compute_metrics(eval_pred):
#     """
#     Compute evaluation metrics for HuggingFace Trainer.

#     Args:
#         eval_pred: tuple (predictions, labels) from model evaluation

#     Returns:
#         dict: accuracy, precision, recall, f1 scores
#     """
#     print("Starting eval")
#     metric1 = evaluate.load("accuracy")
#     metric2 = evaluate.load("recall")
#     metric3 = evaluate.load("precision")
#     metric4 = evaluate.load("f1")

#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=2)  # token-level class predictions

#     # Flatten arrays
#     predictions = np.array(predictions, dtype='int32').flatten()
#     labels = np.array(labels, dtype='int32').flatten()

#     # Replace -100 (ignored tokens) by 0 so metrics do not break
#     labels = [0 if x == -100 else x for x in labels]

#     # Compute metrics
#     acc = metric1.compute(predictions=predictions, references=labels)
#     recall = metric2.compute(predictions=predictions, references=labels, average=None)
#     precision = metric3.compute(predictions=predictions, references=labels, average=None)
#     f1 = metric4.compute(predictions=predictions, references=labels, average=None)

#     # Flatten dicts into lists
#     recall_l = []
#     [recall_l.extend(v) for k, v in recall.items()]
#     precision_l = []
#     [precision_l.extend(v) for k, v in precision.items()]
#     f1_l = []
#     [f1_l.extend(v) for k, v in f1.items()]

#     print("Eval finished")
#     return {
#         "accuracy": acc,
#         "recall": recall_l,
#         "precision": precision_l,
#         "f1": f1_l
#     }
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # ⚠️ Important : filtrer les labels ignorés (-100)
    labels = labels.flatten()
    predictions = predictions.flatten()
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
