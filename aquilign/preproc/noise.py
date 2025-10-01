# -*- coding: utf-8 -*-
"""
noise.py
--------
Utility functions for injecting artificial "noise" into texts,
to simulate errors often found in HTR (Handwritten Text Recognition)
outputs or semi-diplomatic transcriptions.

Goal:
    Make the segmentation model more robust by exposing it to
    less regularized, noisier inputs.

Main entry point:
    apply_noise(text, labels=None, noise_level="medium")
"""

import random
import re


# -------------------------------------------------------------------
# Character confusion map (simulate typical OCR/HTR confusions)
# -------------------------------------------------------------------
CONFUSION_MAP = {
    "u": "v", "v": "u",
    "i": "l", "l": "i",
    "m": "n", "n": "m",
    "c": "e", "e": "c",
    "t": "c", "s": "f"
}


def char_confusion(word, prob=0.1):
    """
    Replace characters in a word with common confusions.
    
    prob (float): probability between 0 and 1
        - prob=0.1 → each character has a 10% chance of being replaced
        - higher values → stronger corruption

    Example:
        "dominus" -> "dominvs"
    """
    return "".join(
        CONFUSION_MAP.get(c, c) if random.random() < prob else c
        for c in word
    )


def random_delete_char(word, prob=0.1):
    """
    Randomly delete characters in a word.

    prob (float): probability each character is dropped.

    Example:
        "dominus" -> "domins"
    """
    if len(word) <= 1:
        return word
    return "".join([c for c in word if random.random() > prob])


def drop_punctuation(text, prob=0.3):
    """
    Randomly remove punctuation marks from the text.

    prob (float): probability each punctuation mark is deleted.

    Example:
        "In principio, Deus." -> "In principio Deus"
    """
    return re.sub(r"([.,;:!?])",
                  lambda m: "" if random.random() < prob else m.group(1),
                  text)


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def apply_noise(text, labels=None, noise_level="medium"):
    """
    Apply a configurable set of noise transformations to text.

    Args:
        text (str): input text
        labels (list): segmentation labels (passed through unchanged)
        noise_level (str): 'light' | 'medium' | 'heavy'
            Controls the intensity of the noise:
                - light  = few modifications
                - medium = balanced corruption
                - heavy  = strong corruption

    Returns:
        (noisy_text, labels): noisy text string + unchanged labels
    """

    # Define probabilities depending on noise level
    if noise_level == "light":
        p_del, p_conf, p_punct = 0.05, 0.05, 0.1
    elif noise_level == "medium":
        p_del, p_conf, p_punct = 0.1, 0.1, 0.3
    else:  # heavy
        p_del, p_conf, p_punct = 0.2, 0.2, 0.5

    # Apply word-level noise
    noisy_words = []
    for w in text.split():
        w = random_delete_char(w, prob=p_del)
        w = char_confusion(w, prob=p_conf)
        noisy_words.append(w)

    noisy_text = " ".join(noisy_words)

    # Apply punctuation noise
    noisy_text = drop_punctuation(noisy_text, prob=p_punct)

     # Debug print to check when noise is applied
    print("[NOISE APPLIED]", noisy_text)

    return noisy_text, labels


# -------------------------------------------------------------------
# OPTIONAL FUTURE EXTENSION:
# If you want fine-grained control instead of only presets,
# change the function signature to:
#
# def apply_noise(text, labels=None, noise_level="medium",
#                 p_del=None, p_conf=None, p_punct=None):
#
# Then override defaults like this:
# p_del = p_del if p_del is not None else p_del_default
# ...
#
# This way you can call:
# apply_noise(text, p_del=0.2, p_conf=0.05, p_punct=0.4)
# -------------------------------------------------------------------
