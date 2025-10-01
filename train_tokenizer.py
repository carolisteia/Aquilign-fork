# -*- coding: utf-8 -*-
"""
train_tokenizer.py
------------------
Script for training a sentence boundary detection model using BERT.

Steps:
  1. Load JSON corpora (train/dev/eval)
  2. Optionally apply artificial noise (HTR-like)
  3. Tokenize and align labels
  4. Train with HuggingFace Trainer
  5. Evaluate and save best checkpoint
"""

import os
import argparse
import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from aquilign.preproc import utils
from aquilign.preproc.tok_trainer_functions import (
    SentenceBoundaryDataset,
    compute_metrics,
    LANG_NOISE_CONFIG,
)


# -------------------------------------------------------------------
# Training wrapper
# -------------------------------------------------------------------
def training_trainer(
    modelName,
    outName,
    train_texts_and_labels,
    dev_texts_and_labels,
    eval_texts_and_labels,
    device="cpu",
    batch_size=32,
    epochs=10,
    save_every=1,
    logging_steps=500,
    early_stopping=8,
):
    """
    Main function to configure and launch HuggingFace Trainer.
    """

    # Load pretrained BERT model
    model = AutoModelForTokenClassification.from_pretrained(modelName, num_labels=3)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"results_{outName}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",  # no wandb/logging
    )

    # Build datasets
    train_dataset = SentenceBoundaryDataset(train_texts_and_labels, tokenizer, lang=args.lang, debug_noise=args.debug_noise)
    dev_dataset = SentenceBoundaryDataset(dev_texts_and_labels, tokenizer, lang=args.lang)
    eval_dataset = SentenceBoundaryDataset(eval_texts_and_labels, tokenizer, lang=args.lang)

    # HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save best model
    best_path = os.path.join(training_args.output_dir, "best")
    trainer.save_model(best_path)
    print(f"\nBest model can be found at : {best_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
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

    # Noise/debug
    parser.add_argument("--noise", action="store_true", help="Enable artificial noise on training set")
    parser.add_argument("--noise_prob", type=float, default=None, help="Override default noise probability")
    parser.add_argument("--noise_level", type=str, default=None, help="Override noise level: light|medium|heavy")
    parser.add_argument("--debug_noise", action="store_true", help="Print up to 5 noisy examples per language")
    parser.add_argument("--lang", type=str, required=True, help="Language code (fr, pt, la, es, it, ca, en)")

    args = parser.parse_args()
    use_cpu = (args.device == "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # -------------------------------------------------------------------
    # Load corpora
    # -------------------------------------------------------------------
    # Always clean for dev/eval
    dev_lines = utils.json_corpus_to_lines(args.dev_dataset, keep_punct=True)
    eval_lines, delimiter = utils.json_corpus_to_lines(args.eval_dataset, keep_punct=True, return_delimiter=True)

    # Training corpus with optional noise
    if args.noise:
        # Get defaults from LANG_NOISE_CONFIG
        cfg = LANG_NOISE_CONFIG.get(args.lang, {"noise_prob": 0.3, "noise_level": "medium"})
        noise_prob = args.noise_prob if args.noise_prob is not None else cfg["noise_prob"]
        noise_level = args.noise_level if args.noise_level is not None else cfg["noise_level"]

        train_lines = utils.json_corpus_to_lines(
            args.train_dataset,
            keep_punct=True,
            apply_noise_flag=True,
            noise_prob=noise_prob,
            noise_level=noise_level,
            debug_noise=args.debug_noise,
        )
    else:
        train_lines = utils.json_corpus_to_lines(args.train_dataset, keep_punct=True)

    # Extract raw texts
    train_texts = [e["example"] for e in train_lines]
    dev_texts = [e["example"] for e in dev_lines]
    eval_texts = [e["example"] for e in eval_lines]

    # -------------------------------------------------------------------
    # Convert to tokenized + labels
    # -------------------------------------------------------------------
    train_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(train_texts, tokenizer=tokenizer, delimiter=delimiter)
    dev_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(dev_texts, tokenizer=tokenizer, delimiter=delimiter)
    eval_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(eval_texts, tokenizer=tokenizer, delimiter=delimiter)

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    training_trainer(
        modelName=args.model,
        outName=args.out_name,
        train_texts_and_labels=train_texts_and_labels,
        dev_texts_and_labels=dev_texts_and_labels,
        eval_texts_and_labels=eval_texts_and_labels,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every=args.save_every,
        logging_steps=args.logging_steps,
        early_stopping=args.early_stopping,
    )
