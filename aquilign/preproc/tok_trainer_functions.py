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

import re
import torch
import evaluate
import numpy as np
import tqdm 
from torch.utils.data import Dataset
import random
from aquilign.preproc.noise import apply_noise


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

    - Loads (text, labels) pairs
    - Optionally applies noise to simulate HTR-like input
    - Tokenizes text with BERT tokenizer
    - Aligns segmentation labels to subword tokens

    Args:
        texts_and_labels (list): list of (text, labels) tuples
            - text (str): input string (a sentence or passage)
            - labels (list of int): gold segmentation labels
                - 0 = no boundary
                - 1 = boundary
        tokenizer: HuggingFace tokenizer
        apply_noise_flag (bool): if True, activate noise injection
        noise_prob (float): probability of applying noise to an example
        noise_level (str): 'light' | 'medium' | 'heavy'
    """

    

    def __init__(self, texts_and_labels, tokenizer,
                 apply_noise_flag=False, noise_prob=0.3, noise_level="medium", debug_noise=False):
        self.data = texts_and_labels
        self.tokenizer = tokenizer
        self.apply_noise_flag = apply_noise_flag
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.debug_noise = debug_noise   
        self.debug_counter = 0           # limit how many times we print


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels = self.data[idx]

    # -----------------------------------------------------------
        # NEW STEP: optional noise injection
        # - Controlled by apply_noise_flag
        # - Only some samples are noised, with probability noise_prob
        # - Uses apply_noise() from aquilign/preproc/noise.py
        # -----------------------------------------------------------

        # Apply noise with given probability
        if self.apply_noise_flag and random.random() < self.noise_prob:
            text, labels = apply_noise(text, labels, noise_level=self.noise_level)
        # Debug printing: show only first 5 noisy samples
            if self.debug_noise and self.debug_counter < 5:
                print("[NOISE DEBUG]", text)
                self.debug_counter += 1

    # -----------------------------------------------------------
        # Tokenization
        # - Converts text into subword tokens compatible with BERT
        # - truncation/padding ensures fixed length (max_length=128)
        # - return_tensors='pt' gives PyTorch tensors
        # -----------------------------------------------------------

        # Tokenize with HuggingFace tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,   # fixed max length, can be tuned
            return_tensors="pt",
            is_split_into_words=False
        )

    # -----------------------------------------------------------
        # Label alignment
        # - HuggingFace tokenizer splits words into subwords
        # - We need to expand labels from word-level to subword-level
        # - word_ids gives mapping between tokens and original words
        # - Special tokens ([CLS], [SEP]) get label -100 to be ignored
        # -----------------------------------------------------------

        # Align labels to tokenized sequence
        word_ids = encoding.word_ids(batch_index=0)
        token_labels = []
        for word_id in word_ids:
            if word_id is None:
                token_labels.append(-100)  # ignored by loss function
            else:
                token_labels.append(labels[word_id])
        encoding["labels"] = token_labels

    # -----------------------------------------------------------
        # HuggingFace Trainer expects dict of tensors
        # - .squeeze(0) removes the extra batch dimension
        # -----------------------------------------------------------

        # Squeeze to remove extra batch dimension
        return {key: val.squeeze(0) for key, val in encoding.items()}


# -------------------------------------------------------------------
# Metrics computation
# -------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for HuggingFace Trainer.

    Args:
        eval_pred: tuple (predictions, labels) from model evaluation

    Returns:
        dict: accuracy, precision, recall, f1 scores
    """
    print("Starting eval")
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("precision")
    metric4 = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)  # token-level class predictions

    # Flatten arrays
    predictions = np.array(predictions, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()

    # Replace -100 (ignored tokens) by 0 so metrics do not break
    labels = [0 if x == -100 else x for x in labels]

    # Compute metrics
    acc = metric1.compute(predictions=predictions, references=labels)
    recall = metric2.compute(predictions=predictions, references=labels, average=None)
    precision = metric3.compute(predictions=predictions, references=labels, average=None)
    f1 = metric4.compute(predictions=predictions, references=labels, average=None)

    # Flatten dicts into lists
    recall_l = []
    [recall_l.extend(v) for k, v in recall.items()]
    precision_l = []
    [precision_l.extend(v) for k, v in precision.items()]
    f1_l = []
    [f1_l.extend(v) for k, v in f1.items()]

    print("Eval finished")
    return {
        "accuracy": acc,
        "recall": recall_l,
        "precision": precision_l,
        "f1": f1_l
    }
