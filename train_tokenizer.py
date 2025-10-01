# -*- coding: utf-8 -*-
"""
train_tokenizer.py
------------------
Main training script for clause/sentence segmentation with BERT.

Overview
--------
This script fine-tunes a pretrained BERT (or compatible Transformer model)
for the task of clause or sentence boundary detection. It casts the problem
as token classification: each token is labeled either as
    0 = no boundary
    1 = boundary

The training pipeline includes:
    - Loading datasets (train/dev/eval) from JSON files.
      Each dataset must provide text strings plus gold boundary labels.
    - Tokenizing input strings into subwords (BERT WordPiece).
    - Expanding/aligning word-level segmentation labels to subword-level labels.
    - Fine-tuning a pretrained model for token classification.
    - Evaluating on dev and eval sets at each epoch.
    - Selecting the best checkpoint (by precision) and renaming it "best".

Noise augmentation
------------------
By default, no noise is applied (clean training only).  
To enable noise augmentation, add the flag:

    --noise --noise_prob 0.3 --noise_level medium

Noise is applied **only to the training set**.  
Dev and eval datasets always remain clean.

Outputs
-------
The script writes:
    - Model checkpoints after each epoch
    - Logs and metrics for each step/epoch
    - A "best" checkpoint directory containing the model with highest precision
    - Evaluation results on the eval set
"""



"""
train_tokenizer.py
-------------------
Train a segmentation model with HuggingFace Transformers.

Steps:
  - Load corpora (JSON format)
  - Optionally apply noise augmentation for training
  - Tokenize and prepare datasets
  - Train with HuggingFace Trainer
  - Evaluate model performance
"""

import os
import json
import random
import argparse
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)

import evaluate

# Local imports
import aquilign.preproc.utils as utils
import aquilign.preproc.tok_trainer_functions as tok_utils

from aquilign.preproc.tok_trainer_functions import (
    SentenceBoundaryDataset,
    compute_metrics,
    LANG_NOISE_CONFIG
)


# -------------------------------------------------------------------
# Custom callback: save every N epochs
# -------------------------------------------------------------------
class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every):
        super().__init__()
        self.save_every = save_every

    def on_epoch_end(self, args, state, control, **kwargs):
        if int(state.epoch) % self.save_every == 0:
            control.should_save = True
        else:
            control.should_save = False


# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------
def training_trainer(
    modelName,
    train_dataset,
    dev_dataset,
    eval_dataset,
    num_train_epochs,
    batch_size,
    logging_steps,
    use_cpu,
    bf_16,
    out_name,
    save_every,
    early_stopping,
    apply_noise,
    noise_prob,
    noise_level,
    debug_noise,
):

    print("Preparing train corpus...")

    # Always clean for dev/eval
    dev_lines = utils.json_corpus_to_lines(dev_dataset, keep_punct=True)
    eval_lines, delimiter = utils.json_corpus_to_lines(
        eval_dataset, keep_punct=True, return_delimiter=True
    )

    # Training corpus: always clean here
    train_lines = utils.json_corpus_to_lines(train_dataset, keep_punct=True)

    # # Extract raw texts
    # train_texts = [e["example"] for e in train_lines]
    # dev_texts = [e["example"] for e in dev_lines]
    # eval_texts = [e["example"] for e in eval_lines]

    train_texts = train_lines
    dev_texts = dev_lines
    eval_texts = eval_lines

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    # Convert to subwords & labels
    train_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(
        train_texts, tokenizer=tokenizer, delimiter=delimiter
    )
    dev_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(
        dev_texts, tokenizer=tokenizer, delimiter=delimiter
    )
    eval_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(
        eval_texts, tokenizer=tokenizer, delimiter=delimiter
    )

    # HuggingFace Datasets
    train_dataset = SentenceBoundaryDataset(
        train_texts_and_labels,
        tokenizer,
        lang="fr",  # for now fixed, but args.lang can be passed
        debug_noise=debug_noise
    )
    dev_dataset = SentenceBoundaryDataset(
        dev_texts_and_labels,
        tokenizer,
        lang="fr",
        debug_noise=False
    )
    eval_dataset = SentenceBoundaryDataset(
        eval_texts_and_labels,
        tokenizer,
        lang="fr",
        debug_noise=False
    )

    # Model
    model = AutoModelForTokenClassification.from_pretrained(modelName, num_labels=3)

    # Training args
    training_args = TrainingArguments(
        output_dir=f"results_{out_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        bf16=bf_16,
        no_cuda=use_cpu,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            SaveEveryNEpochsCallback(save_every=save_every),
            EarlyStoppingCallback(early_stopping_patience=early_stopping)
        ],
    )

    # Train
    trainer.train()

    print("End of training")

    # Final eval
    print("Performing regexp based tokenization evaluation")
    evaluation_synt = utils.evaluate_tokenization_with_regexp(eval_lines, delimiter)
    print(evaluation_synt)

    print("Performing bert-based tokenization evaluation")
    evaluation_bert = utils.evaluate_tokenization_with_bert(
        trainer, eval_dataset, tokenizer, delimiter
    )
    print(evaluation_bert)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import set_seed

    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Base pretrained model to fine-tune")
    parser.add_argument("-n", "--out_name", required=True, help="Experiment name / output dir")
    parser.add_argument("-t", "--train_dataset", required=True, help="Path to train dataset (JSON)")
    parser.add_argument("-d", "--dev_dataset", required=True, help="Path to dev dataset (JSON)")
    parser.add_argument("-e", "--eval_dataset", required=True, help="Path to eval dataset (JSON)")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("-l", "--logging_steps", type=int, default=500, help="Logging frequency")
    parser.add_argument("-es", "--early_stopping", type=int, default=8, help="Early stopping patience")
    parser.add_argument("-dev", "--device", default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("-s", "--save_every", type=int, default=1, help="Save every N epochs")
    parser.add_argument("-bf16", "--bfloat16", action=argparse.BooleanOptionalAction, default=False,
                        help="Use bfloat16 precision if supported")

    # Noise-related arguments
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Apply noise augmentation to training dataset (default: OFF). "
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.3,
        help="Probability of applying noise to a training example (only used if --noise)."
    )
    parser.add_argument(
        "--noise_level",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Noise intensity level (only used if --noise)."
    )
    parser.add_argument(
        "--debug_noise",
        action="store_true",
        help="Debug mode: print up to 5 noisy examples to check augmentation."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language code for the dataset (e.g. fr, pt, la, es, it, ca, en)."
    )

    args = parser.parse_args()
    use_cpu = (args.device == "cpu")

    training_trainer(
        modelName=args.model,
        train_dataset=args.train_dataset,
        dev_dataset=args.dev_dataset,
        eval_dataset=args.eval_dataset,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        use_cpu=use_cpu,
        bf_16=args.bfloat16,
        out_name=args.out_name,
        save_every=args.save_every,
        early_stopping=args.early_stopping,
        apply_noise=args.noise,
        noise_prob=args.noise_prob,
        noise_level=args.noise_level,
        debug_noise=args.debug_noise,
    )
