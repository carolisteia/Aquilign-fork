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
Noise is handled *per language*, via the LANG_NOISE_CONFIG dictionary
(see tok_trainer_functions.py).  
By default, each language has its own noise settings:
    - e.g. French gets heavier noise, Portuguese very light noise, etc.

Dev and eval datasets always remain clean.

Outputs
-------
The script writes:
    - Model checkpoints after each epoch
    - Logs and metrics for each step/epoch
    - A "best" checkpoint directory containing the model with highest precision
    - Evaluation results on the eval set

Usage example
-------------
Train clean French model:
    python train_tokenizer.py \
        -m bert-base-multilingual-cased \
        -n experiment_fr \
        -t data/fr/train.json \
        -d data/fr/dev.json \
        -e data/fr/eval.json \
        -ep 10 -b 8 \
        --lang fr

Train clean Portuguese model:
    python train_tokenizer.py \
        -m bert-base-multilingual-cased \
        -n experiment_pt \
        -t data/pt/train.json \
        -d data/pt/dev.json \
        -e data/pt/eval.json \
        -ep 10 -b 8 \
        --lang pt
"""

import sys
import re
import os
import json
import glob
import argparse
import jsonschema

from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForTokenClassification,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback,
)

# Project-specific modules
import aquilign.preproc.tok_trainer_functions as trainer_functions
import aquilign.preproc.eval as evaluation
import aquilign.preproc.utils as utils


# -------------------------------------------------------------------
# Custom callback: save every N epochs
# -------------------------------------------------------------------
class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every):
        self.save_every = save_every

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.save_every == 0:
            control.should_save = True
        else:
            control.should_save = False


# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------
def training_trainer(modelName,
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
                     keep_punct=True,
                     debug_noise=False):  # print up to 5 noisy examples per language
    """
    Train and evaluate a BERT model for segmentation.

    Args:
        modelName (str): pretrained model name/path
        train_dataset/dev_dataset/eval_dataset (str): paths to JSON files
        num_train_epochs (int): number of epochs
        batch_size (int): batch size per device
        logging_steps (int): logging frequency
        use_cpu (bool): force CPU usage
        bf_16 (bool): use bfloat16 precision
        out_name (str): experiment name
        save_every (int): save every N epochs
        early_stopping (int): early stopping patience
        keep_punct (bool): keep punctuation in preprocessing
        debug_noise (bool): if True, print some noisy samples
    """

    # Load corpora
    train_lines = utils.json_corpus_to_lines(train_dataset, keep_punct)
    dev_lines = utils.json_corpus_to_lines(dev_dataset, keep_punct)
    eval_lines, delimiter = utils.json_corpus_to_lines(
        eval_dataset, keep_punct, return_delimiter=True
    )
    eval_data_lang = eval_dataset.split("/")[-2]

    # Model + tokenizer
    model = AutoModelForTokenClassification.from_pretrained(modelName, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(modelName, max_length=10)

    # Prepare datasets
    print("Train corpus preparation")
    train_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(
        train_lines, tokenizer=tokenizer, delimiter=delimiter
    )
    train_dataset = trainer_functions.SentenceBoundaryDataset(
        train_texts_and_labels,
        tokenizer,
        lang=args.lang,              # language-specific noise config
        debug_noise=debug_noise
    )

    print("Dev corpus preparation")
    dev_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(
        dev_lines, tokenizer=tokenizer, delimiter=delimiter
    )
    dev_dataset = trainer_functions.SentenceBoundaryDataset(
        dev_texts_and_labels,
        tokenizer,
        lang=args.lang               # dev set stays clean
    )

    # HuggingFace training args
    training_args = TrainingArguments(
        output_dir=f"results_{out_name}/epoch{num_train_epochs}_bs{batch_size}",
        num_train_epochs=num_train_epochs,
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        bf16=bf_16,
        use_cpu=use_cpu,
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=trainer_functions.compute_metrics,
        callbacks=[
            SaveEveryNEpochsCallback(save_every=save_every),
            EarlyStoppingCallback(early_stopping_patience=early_stopping)
        ]
    )

    # Evaluate before fine-tuning
    print("Evaluating model before finetuning.")
    evaluation.run_eval(
        data=eval_lines,
        model_path=modelName,
        tokenizer_name=modelName,
        verbose=False,
        delimiter=delimiter
    )

    # Train
    print("Starting training")
    trainer.train()
    print("End of training")

    # Best checkpoint selection
    best_precision_step, best_step_metrics = utils.get_best_step(trainer.state.log_history)
    all_checkpoints = glob.glob(f"results_{out_name}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-*")
    as_ints = [int(path.split("-")[-1]) for path in all_checkpoints]
    all_diffs = [abs(best_precision_step - ckpt) for ckpt in as_ints]
    best_model_path = all_checkpoints[all_diffs.index(min(all_diffs))]

    print(f"Best model path according to precision: {best_model_path}")
    print(f"Full metrics: {best_step_metrics}")

    # Final eval
    eval_results = evaluation.run_eval(
        data=eval_lines,
        model_path=best_model_path,
        tokenizer_name=tokenizer.name_or_path,
        verbose=False,
        delimiter=delimiter
    )

    # Rename best checkpoint
    new_best_path = f"results_{out_name}/epoch{num_train_epochs}_bs{batch_size}/best"
    try:
        os.rmdir(new_best_path)
    except FileNotFoundError:
        pass
    os.rename(best_model_path, new_best_path)

    # Save metadata
    with open(f"{new_best_path}/model_name", "w") as model_name_file:
        model_name_file.write(modelName)
    with open(f"{new_best_path}/eval.txt", "w") as evaluation_results:
        evaluation_results.write(eval_results)
    with open(f"{new_best_path}/metrics.json", "w") as metrics_file:
        json.dump(best_step_metrics, metrics_file)

    print(f"\n\nBest model can be found at : {new_best_path}")
    print(f"Cleanup tip: remove extra checkpoints with\n"
          f"   rm -r results_{out_name}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-*")

    return new_best_path


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == '__main__':
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

        # Noise-related arguments (default OFF)
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Apply noise augmentation to training dataset (default: OFF). "
             "Use --noise to enable. Dev and eval sets are always kept clean."
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
        help="Debug mode: print up to 5 noisy examples per language."
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
        debug_noise=args.debug_noise
    )
