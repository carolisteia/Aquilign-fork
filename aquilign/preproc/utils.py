# -*- coding: utf-8 -*-
"""
utils.py
--------
Utility functions for loading and preparing corpora
for sentence/clause segmentation training.

Changes:
--------
- Added optional noise injection at corpus loading stage (train only).
- Controlled by arguments: apply_noise_flag, noise_prob, noise_level.
- Uses apply_noise() from aquilign.preproc.noise.

"""

import re
import torch
import jsonschema
import json
import unicodedata
import random
import aquilign.preproc.tok_trainer_functions as functions
from aquilign.preproc.noise import apply_noise


def tokenize(text,num):
    words = text.split(" ")
    return [' '.join(words[i:i+num]) for i in range(0, len(words), num)]


def get_best_step(results):
    """
    This function gets the best metrics of label 1 (= delimiter) given the results of the trainer.
    As for now it is the weighted average of precision (w=2) and recall (w=1) 
    """
    print(results)
    result_dict = {}
    for result in results:
        try:
            result_dict[result['step']] = {**result_dict[result['step']], **result}
        except KeyError:
            result_dict[result['step']] = result

    all_metrics = {}
    for key, value in result_dict.items():
        metric = (value['eval_precision'][1] + value['eval_recall'][1]*2)/3
        all_metrics[key] = metric

    best_step = next(step for step, metric in all_metrics.items() if metric == max(all_metrics.values()))
    print(f"Best step according to precision: {best_step}")
    return best_step, result_dict[best_step]

def remove_punctuation(text:str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text
    


def tokenize_words(sentence:str, delimiter) -> list:
    """
    Cette fonction tokénise une phrase selon un certain nombre de marqueurs
    """
    words_delimiters = re.compile(r"[\.,;—:\?!’'«»“/\-]|[^\.,;—:\?!’'«»“/\-\s]+")
    sentenceAsList = re.findall(words_delimiters, sentence)
    if delimiter in sentenceAsList:
        # Some workaround for when the delimiter is used on a token in the list of word delimiters.
        alone_delim_index = next(idx for idx, token in enumerate(sentenceAsList) if token == delimiter)
        try:
            to_merge = sentenceAsList.pop(alone_delim_index + 1)
        except IndexError:
            print(f"Index error on sentence:\n '{sentence}'")
            if sentence[-1] == delimiter:
                print("Last char of the sentence should not be the delimiter. Exiting")
            exit(0)
        sentenceAsList[alone_delim_index] = delimiter + to_merge
    return sentenceAsList


def remove_punctuation_from_corpus(data:dict)-> dict:
    """
    This function removes the punctuation from the json formated corpus.
    """
    updated_list_of_examples = []
    for example in data["examples"]:
        without_punct = remove_punctuation(example["example"])
        new_example = {"example": without_punct,
                       "lang": example["lang"]}
        updated_list_of_examples.append(new_example)
    data["examples"] = updated_list_of_examples
    return data


def unicode_normalize_string(string:str) -> str:
    return unicodedata.normalize("NFC", string)


def unicode_normalize_corpus(data:list) -> str:
    normalized_examples = []
    for example in data["examples"]:
        normalized_examples.append({"example": unicode_normalize_string(example["example"]), 
                                    "lang": example["lang"]})
    data["examples"] = normalized_examples
    return data

# def json_corpus_to_lines(corpus:str, keep_punct, return_delimiter=False)-> list[dict]:
#     """
#     This function imports the json files and performs a first validation of the data structure. It returns
#     the examples as a liste of dictionnaries with the example and its language information
#     """
#     with open(corpus, "r") as corpus_file:
#         examples = json.load(corpus_file)
#         if keep_punct is False:
#             examples = remove_punctuation_from_corpus(examples)
    
#     examples = unicode_normalize_corpus(examples)
#     # Let's perform some tests        
#     with open("aquilign/tokenizer/dataSchema.json", "r") as input_file: 
#         JsonSchema = json.load(input_file)
#     test_data(examples, corpus, schema=JsonSchema)
    
#     if return_delimiter:
#         return examples["examples"], examples["metadata"]["delimiter"]
#     else:
#         return examples["examples"]



# # -------------------------------------------------------------------
# # JSON corpus loader
# # -------------------------------------------------------------------
# def json_corpus_to_lines(
#     path,
#     keep_punct=True,
#     return_delimiter=False,
#     apply_noise_flag=False,
#     noise_prob=0.3,
#     noise_level="medium",
#     debug_noise=False,
# ):
#     """
#     Load a JSON corpus and convert to list of raw text examples.

#     Args
#     ----
#     path (str): path to dataset JSON
#     keep_punct (bool): whether to preserve punctuation (not used here)
#     return_delimiter (bool): if True, return (examples, delimiter)
#     apply_noise_flag (bool): if True, apply noise to some examples
#     noise_prob (float): probability of applying noise to an example
#     noise_level (str): noise intensity: 'light' | 'medium' | 'heavy'
#     debug_noise (bool): print noisy samples for debugging

#     Returns
#     -------
#     examples (list[str]) or (examples, delimiter)
#     """

#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     delimiter = data.get("delimiter", "£")
#     examples = []

#     for entry in data["examples"]:
#         text = entry["example"]
#         lang = entry.get("lang", "unk")

#         # -----------------------------------------------------------
#         # NEW: Noise injection BEFORE tokenization
#         # Only applied if apply_noise_flag=True
#         # Each example has prob=noise_prob of being noised
#         # -----------------------------------------------------------
#         if apply_noise_flag and random.random() < noise_prob:
#             text, _ = apply_noise(text, None, noise_level=noise_level)

#             # Debug: print a truncated version of noisy sample
#             if debug_noise:
#                 print(
#                     f"[NOISE DEBUG: {lang.upper()} | prob={noise_prob} | level={noise_level}] "
#                     f"{text[:150]}..."
#                 )

#         examples.append(text)

#     if return_delimiter:
#         return examples, delimiter
#     return examples

# -------------------------------------------------------------------
# JSON corpus loader
# -------------------------------------------------------------------
def json_corpus_to_lines(corpus: str, keep_punct, return_delimiter: bool = False):
    """
    Load JSON corpus, validate schema, and return examples.

    Args
    ----
    corpus (str): path to dataset JSON
    keep_punct (bool): whether to preserve punctuation
    return_delimiter (bool): if True, return (examples, delimiter)

    Returns
    -------
    list[dict] or (list[dict], str)
    Each dict has at least: {"example": <str>, "lang": <str>}
    """
    with open(corpus, "r", encoding="utf-8") as corpus_file:
        examples = json.load(corpus_file)

        if keep_punct is False:
            examples = remove_punctuation_from_corpus(examples)

    examples = unicode_normalize_corpus(examples)

    # Validate with schema
    with open("aquilign/tokenizer/dataSchema.json", "r") as input_file:
        JsonSchema = json.load(input_file)
    test_data(examples, corpus, schema=JsonSchema)

    if return_delimiter:
        return examples["examples"], examples["metadata"]["delimiter"]
    else:
        return examples["examples"]


def test_data(data:dict, label:str, schema:dict) -> None:
    """
    This function tests if the training data can be correctly parsed. If not, it stops the training and exits.
    """
    # We first test if the data format is OK
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        print(f"The data is not valid. Please make sure the structure follows the example in the README. "
              f"Error: {e}")
        exit(0)
        
    delimiter = data['metadata']["delimiter"]
    regexp = re.compile(rf"{delimiter}([^A-Za-zẽ\d+çÇÉÁÍòãÓȝïÈèÚéçáíƷàÞóúýþ&\(\)\[\].·,,;¿?¦“…/’‘>«»'¡\-—–―\"])\s?")
    valid_list = []
    for idx, example in enumerate(data["examples"]):
        example_text = example["example"]
        search = re.search(regexp, example_text)
        if search:
            print("\n")
            print(f"Problem with some example (example {idx + 1}):\n{example_text}")
            print(search)
            print("\n")
            valid_list.append(False)
    
    if any([item is False for item in valid_list]):
        print(f"Test on {label} failed. Exiting")
    else:
        print(f"Test on {label} passed.")
    
    

def convertToWordsSentencesAndLabels(corpus:list[dict|str], delimiter="£") -> (list, list):
    """
    This function take a corpus as a list of examples and returns the masks for each token as words
    """

    sentencesList = []
    sentencesAsLabels = []
    sentences_as_list_of_tokens = []
    langsList = []
    for example in corpus:
        # On peut avoir une liste de dictionnaires ou de chaînes de caractères
        if isinstance(example, dict):
            text = example["example"]
            langsList.append(example["lang"])
        else:
            text = example
        sentenceAsList = tokenize_words(text, delimiter)
        sentences_as_list_of_tokens.append(sentenceAsList)
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append(sentence)
    return sentencesList, sentencesAsLabels, sentences_as_list_of_tokens, langsList


# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertToSubWordsSentencesAndLabels(corpus, tokenizer, delimiter="£",  verbose=False):
    """
    This function takes a corpus and returns the tokenized corpus as subwords with their labels.
    :param corpus: A list of dicts of the shape 
                            {"example": "tutti e tre £e domandarono quali armi il cavaliere ne portò £quand’e'", 
                             "lang": "it"}
    """
    if verbose:
        print("Converting to sentences and labels")
    sentencesList = []
    sentencesAsLabels = []
    for example in corpus:
        text = example["example"]
        sentenceAsList = tokenize_words(text, delimiter)
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append(sentence)
    num_max_length = functions.get_token_max_length(sentencesList, tokenizer)
    out_toks_and_labels = []
    for text, labels in zip(sentencesList, sentencesAsLabels):
        toks = tokenizer(text, padding="max_length", max_length=num_max_length, truncation=True,
                         return_tensors="pt")

        # get the text with the similar splits as for the creation of the data
        tokens = tokenize_words(text, delimiter)
        # get the index correspondences between text and tok text
        corresp = functions.get_index_correspondence(tokens, tokenizer)
        # aligning the label
        new_labels = functions.align_labels(corresp, labels, text)
        # get the length of the tensor
        sq = (toks['input_ids'].squeeze())
        ### insert 2 for in the new_labels in order to get tensors with the same size !
        if len(sq) == len(new_labels):
            pass
        else:
            diff = len(sq) - len(new_labels)
            for elem in range(diff):
                new_labels.append(2)
        assert len(sq) == len(new_labels), f"Mismatch.\n" \
                                           f"Text: {text}\n" \
                                           f"{(sq.tolist())}\n" \
                                           f"{(new_labels)}\n" \
                                           f"sq: {len(sq)}\n" \
                                           f"new labels: {len(new_labels)}"
        # tensorize the new labels
        label = torch.tensor(new_labels)
        out_toks_and_labels.append({'input_ids': toks['input_ids'].squeeze(),
                                    'attention_mask': toks['attention_mask'].squeeze(),
                                    'labels': label})
    return out_toks_and_labels
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def evaluate_tokenization_with_regexp(eval_lines, delimiter="£"):
    """
    Évaluation naïve par regexp : 
    compare la présence du délimiteur dans les exemples à une segmentation attendue.
    """
    y_true = []
    y_pred = []

    for ex in eval_lines:
        text = ex["example"]

        # vérité terrain : les positions avec délimiteur
        true_labels = [1 if delimiter in tok else 0 for tok in tokenize_words(text, delimiter)]
        # prédiction naïve : "toujours pas de délimiteur" (baseline)
        pred_labels = [0] * len(true_labels)

        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "regexp_accuracy": acc,
        "regexp_precision": precision,
        "regexp_recall": recall,
        "regexp_f1": f1,
    }


def evaluate_tokenization_with_bert(trainer, eval_dataset, tokenizer, delimiter="£"):
    """
    Évaluation avec le modèle BERT entraîné :
    on passe l'ensemble d'évaluation au Trainer et on calcule les métriques.
    """
    preds_output = trainer.predict(eval_dataset)
    logits = preds_output.predictions
    labels = preds_output.label_ids

    predictions = np.argmax(logits, axis=-1).flatten()
    labels = labels.flatten()

    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    acc = accuracy_score(labels, predictions)

    return {
        "bert_accuracy": acc,
        "bert_precision": precision,
        "bert_recall": recall,
        "bert_f1": f1,
    }
