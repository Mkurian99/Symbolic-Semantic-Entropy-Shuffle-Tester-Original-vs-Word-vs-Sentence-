"""
================================================================================
SYMBOLIC ENTROPY (SE) MASTER CALCULATOR - Three-Way Comparison Edition
================================================================================

FEATURES:
✓ Layer 1: Core SE Analysis (H + Σ with proper KL divergence)
✓ Layer 2: Peak/Valley Analysis with Text Quotes
✓ Multi-word Phrase Support (semantic token definition)
✓ Multi-format Support (.txt and .docx files)
✓ THREE-WAY Shuffle Validation: Original vs Word-Shuffle vs Sentence-Shuffle

THEORETICAL FOUNDATION:
    SE = (H, Σ)
    
    Where:
        H = Shannon entropy (bits/semantic-token) - lexical unpredictability
        Σ = KL divergence (bits/semantic-token) - motif clustering beyond baseline

KEY INNOVATION - Semantic Token Definition:
    - "One Ring" = 1 semantic token (merged pre-processing)
    - Phrases in motif dictionary are pre-merged: "one ring" → "one_ring"
    - Results in bits per SEMANTIC TOKEN, not word-token
    - Captures linguistic compression naturally

THREE-WAY SHUFFLE VALIDATION:
    - Accepts three input files: original, word-shuffled, sentence-shuffled
    - All heatmaps use ORIGINAL z-axis scale for valid comparison
    - Reveals hierarchical sensitivity to structural destruction:
        * Word shuffle: Destroys ALL structure → Σ collapses maximally
        * Sentence shuffle: Preserves intra-sentence structure → Σ partially preserved
    - Expected collapse ratios:
        * Word shuffle: 10-20x collapse
        * Sentence shuffle: 3-8x collapse (intermediate)

MOTIF DICTIONARY FORMAT:
    motif_dict = {
        'Category Name': {
            'phrases': ['multi word phrase', 'another phrase'],  # Optional
            'words': ['single', 'word', 'tokens']                # Optional
        }
    }
    
    - Phrase-only motifs: Only include 'phrases' key
    - Word-only motifs: Only include 'words' key  
    - Mixed motifs: Include both keys

IMPLEMENTATION STANDARDS:
    - Adaptive window sizing (auto-scales to text length)
    - Default: ~120 windows with 50% overlap
    - To adjust granularity: Change TARGET_WINDOWS (line ~120)
      - Higher value = more windows = finer resolution
      - Lower value = fewer windows = coarser resolution
    - Global baseline (π_k) from full text distribution
    - Whole-word/phrase matching with word boundaries
    - Falsifiable via shuffle testing (Σ → 0 when randomized)

OUTPUTS (Original):
    - <textname>_se_heatmap.png          (dual heatmap: raw density + KL)
    - <textname>_se_timeseries.png       (H and Σ line graphs)
    - <textname>_peaks_valleys.png       (top 3 peaks/valleys with quotes)
    - <textname>_se_results.csv          (full numerical results)
    - <textname>_peaks_valleys_text.csv  (detailed peak/valley excerpts)

OUTPUTS (Word Shuffle):
    - <textname>_word_shuffle_heatmap.png    (with original z-scale)
    - <textname>_word_shuffle_se_results.csv

OUTPUTS (Sentence Shuffle):
    - <textname>_sent_shuffle_heatmap.png    (with original z-scale)
    - <textname>_sent_shuffle_se_results.csv

OUTPUTS (Comparison):
    - <textname>_3way_comparison.png     (side-by-side KL heatmaps)
    - <textname>_validation_summary.csv  (statistical comparison)

DEPENDENCIES:
    - numpy, pandas, matplotlib, scipy
    - python-docx (for .docx support: pip install python-docx)

USAGE:
    python SE_Master_Calculator_3Way.py <original> <word_shuffle> <sentence_shuffle>
    
    Or edit TEXT_FILE, WORD_SHUFFLE_FILE, SENT_SHUFFLE_FILE variables
    
    To adjust window granularity:
    - Edit TARGET_WINDOWS (line ~120)
    - Default: 120 windows
    - Higher = finer resolution (more windows, smaller size)
    - Lower = coarser resolution (fewer windows, larger size)

VERSION: 3.0.0 - Three-way comparison with hierarchical validation
AUTHOR: Kurian, M. (2025)
================================================================================
"""

import re
import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import find_peaks

# Try to import docx for .docx file support
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Default text files (can be overridden via command line)
# Set to None or empty string to skip that condition
TEXT_FILE = "C:/Users/Michael Kurian/Desktop/PhD Applications/genesis_original.txt.txt"
WORD_SHUFFLE_FILE = "C:/Users/Michael Kurian/Desktop/PhD Applications/genesis_shuffled.txt.txt"
SENT_SHUFFLE_FILE = "C:/Users/Michael Kurian/Desktop/Advanced Researches/TXT Files for SE/genesis_sentence_shuffled.txt.txt"

# Random seed for reproducibility
RANDOM_SEED = 42

# Window parameters - ADAPTIVE sizing
# The window size auto-adjusts to text length to produce ~120 windows
TARGET_WINDOWS = 120  # Adjust this to change granularity (higher = more windows)
# Note: Actual window size will be calculated as: total_tokens / (1 + (TARGET_WINDOWS - 1) / 2)

# ============================================================================
# MOTIF DICTIONARY - FELLOWSHIP OF THE RING (with phrases)
# ============================================================================

motif_dict = {
    # ========================================================================
    # TREE / AXIS MUNDI
    # Central axis connecting heaven and earth; knowledge and life
    # ========================================================================
    'Tree/Axis-Mundi': {
        'phrases': [
            'tree of life',           # Theological concept (appears 2x)
            'tree of knowledge',      # Central to Fall narrative (appears 2x)
            'tree of the knowledge',  # Full phrase variant
            'fruit of the tree',      # Action phrase
        ],
        'words': [
            'tree', 'trees', 'fruit', 'seed', 'yielding', 
            'herb', 'grass', 'plant', 'grow', 'grew', 
            'knowledge', 'bearing', 'green', 'ground', 
            'bring', 'brought', 'kind'
        ]
    },
    
    # ========================================================================
    # WATERS / FLOOD / SEA
    # Primordial waters, chaos, and life-giving fluid
    # ========================================================================
    'Waters/Sea': {
        'phrases': [
            'face of the deep',       # Primordial chaos (Genesis 1:2)
            'face of the waters',     # Spirit moving on waters
            'gathering together of the waters', # Separation act
            'divided the waters',     # Creation act
        ],
        'words': [
            'waters', 'water', 'seas', 'sea', 'river', 'deep', 
            'mist', 'rain', 'watered', 'gathered', 'parted', 
            'divided', 'face', 'pison', 'gihon', 'hiddekel', 
            'euphrates', 'havilah', 'ethiopia', 'assyria'
        ]
    },
    
    # ========================================================================
    # LIGHT
    # Divine light, celestial order, and temporal cycles
    # ========================================================================
    'Light': {
        'phrases': [
            'let there be light',     # Fiat lux - first creation (appears 1x)
            'there was light',        # Divine fulfillment
            'the evening and the morning', # Day formula (appears 6x)
            'divided the light',      # Separation act
        ],
        'words': [
            'light', 'lights', 'darkness', 'day', 'night', 
            'evening', 'morning', 'firmament', 'heaven', 'heavens', 
            'stars', 'rule', 'divide', 
            'signs', 'seasons', 'years', 'greater', 'lesser', 
            'first', 'second', 'third', 'fourth', 'fifth', 
            'sixth', 'seventh'
        ]
    },
    
    # ========================================================================
    # SHARPNESS / SWORD / THORN
    # Weapons, painful vegetation, barriers, and divine judgment
    # ========================================================================
    'Sharpness-Sword-Thorn': {
        'phrases': [
            'flaming sword',          # Guardian weapon at Eden's gate (Gen 3:24)
            'thorns also and thistles', # Curse on ground (Gen 3:18)
        ],
        'words': [
            'sword', 'flaming', 'thorns', 'thistles'
        ]
    },
    
    # ========================================================================
    # DEATH / MORTALITY
    # Death, dust, mortality, and return to earth
    # ========================================================================
    'Death/Mortality': {
        'phrases': [
            'shalt surely die',       # Divine warning (Gen 2:17, 3:4)
            'surely die',             # Serpent's contradiction variant
            'unto dust shalt thou return', # Mortality sentence (Gen 3:19)
            'dust shalt thou return', # Shorter variant
            'dust thou art',          # Mortality declaration
            'return unto the ground', # Return to earth
        ],
        'words': [
            'die', 'dust', 'return'
        ]
    },
    
    # ========================================================================
    # EATING / FORBIDDEN FRUIT
    # Consumption, eating, and the forbidden act
    # ========================================================================
    'Eating/Forbidden-Fruit': {
        'phrases': [
            'thou shalt not eat',     # Divine prohibition (Gen 2:17)
            'shalt not eat',          # Prohibition variant
            'ye shall not',           # Prohibition to Eve (Gen 3:3)
            'did eat',                # Fall action (Gen 3:6, 3:12)
            'i did eat',              # Adam's confession
            'she gave me',            # Adam blaming Eve
        ],
        'words': [
            'eat', 'eaten', 'eatest', 'fruit', 'food', 'meat'
        ]
    },
    
    # ========================================================================
    # CURSE / TOIL
    # Divine curse, sorrow, painful labor, and sweat
    # ========================================================================
    'Curse/Toil': {
        'phrases': [
            'cursed is the ground',   # Ground curse (Gen 3:17)
            'because thou hast',      # Judgment formula (Gen 3:14, 3:17)
            'in sorrow',              # Pain/sorrow formula (Gen 3:16, 3:17)
            'sweat of thy face',      # Toil consequence (Gen 3:19)
        ],
        'words': [
            'cursed', 'sorrow', 'sweat', 'till'
        ]
    },
    
    # ========================================================================
    # TRANSGRESSION / DECEPTION
    # Breaking commands, deception, and serpent's lies
    # ========================================================================
    'Transgression/Deception': {
        'phrases': [
            'ye shall be as gods',    # Serpent's promise (Gen 3:5)
            'shall be as gods',       # Variant
            'knowing good and evil',  # Temptation of knowledge (Gen 3:5)
            'the serpent beguiled',   # Eve's explanation (Gen 3:13)
            'serpent beguiled me',    # Confession variant
        ],
        'words': [
            'beguiled', 'subtil'
        ]
    },
    
    # ========================================================================
    # DESIRE / WISDOM
    # Visual temptation, desire for wisdom, and epistemological lust
    # ========================================================================
    'Desire/Wisdom': {
        'phrases': [
            'pleasant to the eyes',   # Visual temptation (Gen 3:6)
            'desired to make one wise', # Desire for wisdom (Gen 3:6)
            'to make one wise',       # Wisdom temptation variant
            'make one wise',          # Shorter variant
            'a tree to be desired',   # Desirability of tree (Gen 3:6)
            'good for food',          # Sensory appeal (Gen 3:6)
        ],
        'words': [
            'desired', 'desire', 'pleasant', 'wise'
        ]
    },
    
    # ========================================================================
    # FEAR / HIDING
    # Post-Fall fear, hiding from God, and psychological shame
    # ========================================================================
    'Fear/Hiding': {
        'phrases': [
            'i was afraid',           # Adam's fear (Gen 3:10)
            'i heard thy voice',      # Hearing with fear (Gen 3:10)
            'heard thy voice',        # Variant
            'i hid myself',           # Adam's hiding (Gen 3:10)
            'hid themselves',         # Couple hiding (Gen 3:8)
            'hid themselves from',    # Full hiding phrase (Gen 3:8)
        ],
        'words': [
            'afraid', 'hid'
        ]
    },
    
    # ========================================================================
    # EXILE / EXPULSION
    # Driving out from Eden, sending forth
    # ========================================================================
    'Exile/Expulsion': {
        'phrases': [
            'drove out the man',      # Expulsion act (Gen 3:24)
            'sent him forth',         # Sending from garden (Gen 3:23)
            'from the garden of eden', # Departure location
            'from the garden',        # Shorter variant
        ],
        'words': [
            'drove', 'sent', 'forth'
        ]
    },
    
    # ========================================================================
    # SERPENT / BEAST
    # Cunning beasts, chaos creatures, and animal life
    # ========================================================================
    'Serpent/Beast': {
        'phrases': [
            'beast of the field',     # Serpent's domain (appears 7x)
            'beast of the earth',     # Creation category
            'fowl of the air',        # Flying creatures (appears 5x)
            'fish of the sea',        # Aquatic creatures
            'living creature',        # General life
            'every living thing',     # Comprehensive life
        ],
        'words': [
            'serpent', 'beast', 'creature', 'creeping', 
            'creepeth', 'cattle', 'fowl', 'living', 'subtil', 
            'field', 'winged', 'moveth', 'whales', 'fly', 
            'fish', 'air', 'life', 'abundantly', 'multiply', 
            'enmity', 'bruise', 'belly', 'cursed', 'beguiled'
        ]
    },
    
    # ========================================================================
    # BRIDEGROOM-BRIDE
    # Marriage, union, human relationship and sexuality
    # ========================================================================
    'Bridegroom-Bride': {
        'phrases': [
            'male and female',        # Creation duality (appears 2x)
            'bone of my bones',       # Adam's recognition (appears 1x)
            'flesh of my flesh',      # Union phrase
            'one flesh',              # Marriage union (appears 1x)
            'help meet',              # Eve's role (appears 2x)
            'man and his wife',       # Marital pair
            'a living soul',          # Human essence (appears 1x)
        ],
        'words': [
            'man', 'woman', 'wife', 'husband', 'adam', 'eve', 
            'male', 'female', 'bone', 'bones', 'flesh', 'cleave', 
            'leave', 'father', 'mother', 'alone', 'help', 'meet', 
            'together', 'sleep', 'slept', 'ribs', 'rib', 'desire', 
            'conception', 'children', 'seed', 'living', 'hearkened', 
            
        ]
    },
    
    # ========================================================================
    # GARDEN / PARADISE
    # Sacred space, Eden, and geographical landmarks
    # ========================================================================
    'Garden/Paradise': {
        'phrases': [
            'garden of eden',         # Paradise location (appears 4x)
            'midst of the garden',    # Central location (appears 2x)
            'out of eden',            # Exile phrase
            'east of the garden',     # Post-exile location
        ],
        'words': [
            'garden', 'eden', 'pleasant', 'midst', 'east', 
            'eastward', 'planted', 'dress', 'keep', 'place', 
            'land', 'gold', 'bdellium', 'onyx', 'stone', 
            'compasseth', 'river', 'heads', 'parted', 'cherubims', 
            'way', 'whole', 'good', 'sight', 'food'
        ]
    },
    
    # ========================================================================
    # CLOTHING / NAKEDNESS
    # Shame, covering, and transformed consciousness
    # ========================================================================
    'Clothing/Nakedness': {
        'phrases': [
            'they were both naked',   # Pre-Fall description
            'the eyes of them both were opened', # Consciousness change
            'were not ashamed',       # Pre-Fall state
            'coats of skins',         # Divine covering
        ],
        'words': [
            'naked', 'clothed', 'coats', 'skins', 'aprons', 
            'fig', 'leaves', 'sewed', 'ashamed', 'opened', 
            'eyes', 'knew', 'hid', 'afraid', 'closed'
        ]
    },
    
    # ========================================================================
    # SACRED NAME / WORD
    # Divine speech, naming, and creative command
    # ========================================================================
    'Sacred-Name/Word': {
        'phrases': [
            'the lord god',           # Divine name (appears 20x)
            'and god said',           # Creation formula (appears 9x)
            'god said, let',          # Command structure (appears 8x)
            'and god saw',            # Divine approval (appears 7x)
            'god saw that',           # Evaluation phrase
            'it was good',            # Divine verdict (appears 7x)
            'in the beginning',       # Opening phrase (appears 1x)
            'image of god',           # Imago Dei (appears 3x)
            'spirit of god',          # Divine presence
            'breath of life',         # Life-giving act (appears 2x)
            'let us make',            # Divine counsel
        ],
        'words': [
            'god', 'lord', 'name', 'names', 'called', 'call', 
            'voice', 'commanded', 'saying', 'spirit', 'blessed', 
            'sanctified', 'created', 'made', 'rested', 'generations', 
            'breathed', 'formed', 'nostrils', 'breath', 'soul', 
            'image', 'likeness', 'dominion', 'replenish', 'subdue', 
            'behold', 'beginning', 'finished', 'host', 'work'
        ]
    },
    
    # ========================================================================
    # GOOD-EVIL / KNOWLEDGE
    # Moral knowledge, epistemological themes, and ethical consciousness
    # ========================================================================
    'Good-Evil/Knowledge': {
        'phrases': [
            'good and evil',          # Central moral concept (appears 4x)
            'knowledge of good',      # Epistemological phrase
        ],
        'words': [
            'good', 'evil', 'knowing', 'knowledge', 'wise', 
            'wisdom', 'eyes', 'opened', 'gods', 'die', 
            'death', 'surely', 'eat', 'eaten', 'touch'
        ]
    },
    
    # ========================================================================
    # COMMAND / OBEDIENCE
    # Divine imperatives, prohibitions, and speech formulas
    # ========================================================================
    'Command/Obedience': {
        'phrases': [
            'god commanded',          # Divine imperative
            'thou shalt not',         # Prohibition formula (appears 3x)
            'thou shalt surely',      # Emphasis formula
            'let there be',           # Creation command (appears 6x)
            'let them have',          # Dominion grant
            'said unto',              # Speech formula (appears 7x)
        ],
        'words': [
            'commanded', 'saying', 'thou', 'shalt', 'mayest', 
            'freely', 'whereof', 'shouldest',
        ]
    },
    
    # ========================================================================
    # TIME / ORDER
    # Temporal structure, cosmological sequence, and creation days
    # ========================================================================
    'Time/Order': {
        'phrases': [
            'in the beginning',       # Temporal origin
            'the first day',          # Creation sequence (appears 6x)
            'the second day',         # Day 2
            'the third day',          # Day 3
            'the fourth day',         # Day 4
            'the fifth day',          # Day 5
            'the sixth day',          # Day 6
            'the seventh day',        # Sabbath (appears 3x)
            'evening and morning',    # Day formula (appears 6x)
        ],
        'words': [
            'day', 'days', 'beginning', 'first', 'second', 
            'third', 'fourth', 'fifth', 'sixth', 'seventh', 
            'evening', 'morning', 'seasons', 'years', 
            'generations', 'finished'
        ]
    }
}


# ============================================================================
# TEXT PREPROCESSING WITH MULTI-WORD PHRASE SUPPORT
# ============================================================================

def extract_text_from_file(file_path):
    """
    Extract text from .txt or .docx file with robust encoding handling.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.docx':
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    
    elif file_ext == '.txt':
        # Try multiple encodings
        encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode file with any of: {encodings}")
    
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Use .txt or .docx")


def merge_phrases_in_text(text, motif_dict):
    """
    Merge multi-word phrases in text into single tokens.
    Example: "tree of life" -> "tree_of_life"
    
    Returns: (modified_text, phrase_count)
    """
    phrase_count = 0
    
    # Extract all phrases from motif dictionary
    all_phrases = []
    for category_data in motif_dict.values():
        if 'phrases' in category_data:
            all_phrases.extend(category_data['phrases'])
    
    # Sort by length (longest first) to handle overlapping phrases
    all_phrases = sorted(set(all_phrases), key=len, reverse=True)
    
    # Replace each phrase with underscored version
    modified_text = text.lower()
    for phrase in all_phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        replacement = phrase.replace(' ', '_')
        matches = len(re.findall(pattern, modified_text))
        if matches > 0:
            modified_text = re.sub(pattern, replacement, modified_text)
            phrase_count += matches
    
    return modified_text, phrase_count


def tokenize_text(text):
    """
    Tokenize text into words (after phrase merging).
    Uses word boundary matching.
    """
    # Extract words (including underscored phrases)
    tokens = re.findall(r'\b[\w]+\b', text.lower())
    return tokens


# ============================================================================
# ADAPTIVE WINDOW SIZING
# ============================================================================

def calculate_adaptive_window_size(total_tokens, target_windows=120):
    """
    Calculate window size to produce approximately target_windows.
    With 50% overlap: n_windows = 1 + (total_tokens - window_size) / step_size
                                = 1 + (total_tokens - window_size) / (window_size/2)
    Solving: window_size ≈ total_tokens / (1 + (target_windows - 1)/2)
    """
    window_size = int(total_tokens / (1 + (target_windows - 1) / 2))
    
    return window_size


# ============================================================================
# SHANNON ENTROPY CALCULATION
# ============================================================================

def calculate_shannon_entropy(tokens):
    """
    Calculate Shannon entropy H for a list of tokens.
    H = -Σ p(x) log₂ p(x)
    Returns bits per token.
    """
    if not tokens:
        return 0.0
    
    freq = Counter(tokens)
    total = len(tokens)
    
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


# ============================================================================
# BASELINE CALCULATION (GLOBAL TEXT DISTRIBUTION)
# ============================================================================

def calculate_global_baseline(tokens, motif_dict):
    """
    Calculate baseline (π_k) as proportion of each motif in full text.
    This represents the text's overall distribution.
    """
    N = len(tokens)
    baseline = {}
    
    for category, category_data in motif_dict.items():
        total_count = 0
        
        # Count words
        if 'words' in category_data:
            for word in category_data['words']:
                total_count += tokens.count(word)
        
        # Count phrases (now merged with underscores)
        if 'phrases' in category_data:
            for phrase in category_data['phrases']:
                merged_phrase = phrase.replace(' ', '_')
                total_count += tokens.count(merged_phrase)
        
        baseline[category] = total_count / N if N > 0 else 0.0
    
    return baseline


# ============================================================================
# SIGMA (Σ) CALCULATION - KL DIVERGENCE
# ============================================================================

def calculate_sigma_kl(observed, baseline, window_size):
    """
    Calculate Σ using proper KL divergence.
    Formula: Σ_KL = Σ_k p_k × log₂(p_k / π_k)
    Returns bits per token.
    """
    sigma_kl = 0.0
    
    for category in observed.keys():
        obs_count = observed[category]
        p_k = obs_count / window_size
        pi_k = baseline[category]
        
        # Only calculate when both probabilities are non-zero
        if p_k > 0 and pi_k > 0:
            sigma_kl += p_k * np.log2(p_k / pi_k)
    
    return max(0.0, sigma_kl)  # KL divergence is non-negative


# ============================================================================
# MOTIF COUNTING FOR WINDOWS
# ============================================================================

def count_motifs_in_window(window_tokens, motif_dict):
    """
    Count occurrences of each motif category in window.
    Returns dict of {category: count}.
    """
    counts = {}
    
    for category, category_data in motif_dict.items():
        count = 0
        
        # Count words
        if 'words' in category_data:
            for word in category_data['words']:
                count += window_tokens.count(word)
        
        # Count merged phrases
        if 'phrases' in category_data:
            for phrase in category_data['phrases']:
                merged_phrase = phrase.replace(' ', '_')
                count += window_tokens.count(merged_phrase)
        
        counts[category] = count
    
    return counts


# ============================================================================
# MAIN SE ANALYSIS FUNCTION
# ============================================================================

def run_se_analysis(text_path, motif_dict):
    """
    Run complete Symbolic Entropy analysis on text.
    Returns: (results_df, raw_densities, kl_contributions, motif_dict, 
              baseline, window_size, total_tokens, tokens)
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {os.path.basename(text_path)}")
    print(f"{'='*70}\n")
    
    # Extract text
    print("Step 1/6: Extracting text...")
    raw_text = extract_text_from_file(text_path)
    
    # Merge multi-word phrases
    print("Step 2/6: Merging multi-word phrases...")
    text, n_phrases = merge_phrases_in_text(raw_text, motif_dict)
    print(f"  → Merged {n_phrases} phrase occurrences")
    
    # Tokenize
    print("Step 3/6: Tokenizing...")
    tokens = tokenize_text(text)
    total_tokens = len(tokens)
    print(f"  → {total_tokens:,} semantic tokens")
    
    # Calculate adaptive window size
    window_size = calculate_adaptive_window_size(total_tokens, TARGET_WINDOWS)
    step_size = window_size // 2  # 50% overlap
    print(f"\nStep 4/6: Calculating adaptive window parameters...")
    print(f"  → Window size: {window_size} tokens")
    print(f"  → Step size: {step_size} tokens (50% overlap)")
    
    # Calculate global baseline
    print("Step 5/6: Computing global baseline...")
    baseline = calculate_global_baseline(tokens, motif_dict)
    
    # Sliding window analysis
    print("Step 6/6: Running sliding window analysis...")
    results = []
    raw_densities = []
    kl_contributions = []
    
    n_windows = 1 + (total_tokens - window_size) // step_size
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        if end > total_tokens:
            break
        
        window_tokens = tokens[start:end]
        
        # Shannon entropy
        H = calculate_shannon_entropy(window_tokens)
        
        # Motif counts
        observed = count_motifs_in_window(window_tokens, motif_dict)
        
        # Sigma (KL divergence)
        sigma = calculate_sigma_kl(observed, baseline, window_size)
        
        # SE = H + Σ
        se = H + sigma
        
        # Store results
        results.append({
            'window_index': i,
            'start_token': start,
            'end_token': end,
            'H': H,
            'Sigma': sigma,
            'SE': se
        })
        
        # Store raw densities (for heatmap)
        raw_densities.append([observed[cat] / window_size for cat in motif_dict.keys()])
        
        # Store KL contributions (for heatmap)
        kl_contribs = []
        for cat in motif_dict.keys():
            p_k = observed[cat] / window_size
            pi_k = baseline[cat]
            if p_k > 0 and pi_k > 0:
                contrib = p_k * np.log2(p_k / pi_k)
            else:
                contrib = 0.0
            kl_contribs.append(contrib)
        kl_contributions.append(kl_contribs)
    
    results_df = pd.DataFrame(results)
    
    print(f"  → Generated {len(results_df)} windows")
    print(f"✓ Analysis complete!\n")
    
    return (results_df, np.array(raw_densities), np.array(kl_contributions), 
            motif_dict, baseline, window_size, total_tokens, tokens)


# ============================================================================
# VISUALIZATION: DUAL HEATMAP
# ============================================================================

def plot_dual_heatmap(results_df, raw_densities, kl_contributions, 
                     motif_dict, output_prefix, z_limits=None, condition_label=None):
    """
    Plot dual heatmap: Raw motif density + KL contributions.
    Includes overlaid line plots of H (cyan) and Σ (white) on the KL heatmap.
    If z_limits provided, use those for color scaling (for shuffle comparison).
    z_limits format: {'density': (vmin, vmax), 'kl': (vmin, vmax)}
    condition_label: 'original', 'word_shuffle', 'sent_shuffle' for filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    window_indices = results_df['window_index'].values
    
    # Normalize H and Σ for overlay (scale to heatmap Y-axis range)
    H_values = results_df['H'].values
    sigma_values = results_df['Sigma'].values
    
    # Min-max normalization to [0, n_categories-1] range
    H_min, H_max = H_values.min(), H_values.max()
    sigma_min, sigma_max = sigma_values.min(), sigma_values.max()
    
    H_scaled = (n_categories - 1) * (1 - (H_values - H_min) / (H_max - H_min + 1e-10))
    sigma_scaled = (n_categories - 1) * (1 - (sigma_values - sigma_min) / (sigma_max - sigma_min + 1e-10))
    
    # LEFT: Raw density heatmap
    if z_limits and 'density' in z_limits:
        vmin_d, vmax_d = z_limits['density']
    else:
        vmin_d, vmax_d = raw_densities.min(), raw_densities.max()
    
    raw_data = raw_densities.T
    im1 = ax1.imshow(raw_data, aspect='auto', cmap='plasma',
                     interpolation='nearest', origin='lower', 
                     vmin=vmin_d, vmax=vmax_d)
    ax1.set_ylim(-0.5, n_categories - 0.5)
    ax1.set_xlabel('Window Index', fontsize=12)
    ax1.set_ylabel('Motif Category', fontsize=12)
    ax1.set_yticks(range(n_categories))
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_title('Method 1: RAW DENSITY\n(Simple frequency counting)', 
                  fontsize=13, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Proportion of Window', fontsize=10)
    
    # RIGHT: KL Divergence with line overlays
    if z_limits and 'kl' in z_limits:
        vmin_k, vmax_k = z_limits['kl']
    else:
        vmin_k, vmax_k = kl_contributions.min(), kl_contributions.max()
    
    kl_data = kl_contributions.T
    im2 = ax2.imshow(kl_data, aspect='auto', cmap='plasma',
                     interpolation='nearest', origin='lower',
                     vmin=vmin_k, vmax=vmax_k)
    
    # Overlay H and Σ lines (thin and transparent to not block heatmap)
    ax2.plot(window_indices, sigma_scaled, color='white', linewidth=1.0, 
             alpha=0.4, label='Σ (white)')
    ax2.plot(window_indices, H_scaled, color='cyan', linewidth=1.0, 
             alpha=0.4, label='H (cyan)')
    
    ax2.set_ylim(-0.5, n_categories - 0.5)
    ax2.set_xlabel('Window Index', fontsize=12)
    ax2.set_ylabel('Motif Category', fontsize=12)
    ax2.set_yticks(range(n_categories))
    ax2.set_yticklabels(categories, fontsize=9)
    ax2.set_title('Method 2: KL DIVERGENCE (Σ_KL)\n(Structural Surprise - Where motifs cluster)', 
                  fontsize=13, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('KL Contribution (bits/token)', fontsize=10)
    
    # Add legend for line overlays
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.8)
    
    # Determine title based on condition
    title_suffix = ""
    if condition_label == 'word_shuffle':
        title_suffix = " - WORD SHUFFLED"
    elif condition_label == 'sent_shuffle':
        title_suffix = " - SENTENCE SHUFFLED"
    
    fig.suptitle(f'Symbolic Entropy Analysis - Dual Method Visualization{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Determine filename based on condition
    if condition_label == 'word_shuffle':
        filename = f'{output_prefix}_word_shuffle_heatmap.png'
    elif condition_label == 'sent_shuffle':
        filename = f'{output_prefix}_sent_shuffle_heatmap.png'
    else:
        filename = f'{output_prefix}_se_heatmap.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()
    
    # Return z_limits for potential reuse
    if not z_limits:
        return {
            'density': (vmin_d, vmax_d),
            'kl': (vmin_k, vmax_k)
        }
    return z_limits


# ============================================================================
# VISUALIZATION: TIME SERIES
# ============================================================================

def plot_timeseries(results_df, output_prefix):
    """
    Plot H and Σ as time series.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Shannon Entropy (H)
    ax1.plot(results_df['window_index'], results_df['H'], 
             color='steelblue', linewidth=1.5, label='Shannon Entropy (H)')
    ax1.fill_between(results_df['window_index'], results_df['H'], 
                     alpha=0.3, color='steelblue')
    ax1.set_ylabel('H (bits/token)', fontsize=12)
    ax1.set_title('Shannon Entropy Over Text', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Sigma (Σ)
    ax2.plot(results_df['window_index'], results_df['Sigma'], 
             color='crimson', linewidth=1.5, label='Sigma (Σ)')
    ax2.fill_between(results_df['window_index'], results_df['Sigma'], 
                     alpha=0.3, color='crimson')
    ax2.set_xlabel('Window Index', fontsize=12)
    ax2.set_ylabel('Σ (bits/token)', fontsize=12)
    ax2.set_title('Sigma (Motif Concentration) Over Text', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    filename = f'{output_prefix}_se_timeseries.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


# ============================================================================
# PEAK/VALLEY DETECTION WITH TEXT EXTRACTION
# ============================================================================

def extract_window_text(window_idx, tokens, window_size, step_size, context_words=50):
    """
    Extract text for a given window with context.
    Returns dict with window text, context, and positions.
    """
    start = window_idx * step_size
    end = start + window_size
    
    # Extract window tokens
    window_tokens = tokens[start:end]
    
    # Extract context (before and after)
    context_start = max(0, start - context_words)
    context_end = min(len(tokens), end + context_words)
    full_context_tokens = tokens[context_start:context_end]
    
    # Join tokens back to text (replace underscores with spaces for readability)
    window_text = ' '.join(window_tokens).replace('_', ' ')
    full_context = ' '.join(full_context_tokens).replace('_', ' ')
    
    return {
        'window_text': window_text,
        'full_context': full_context,
        'start_position': start,
        'end_position': end
    }


def analyze_window_motifs(window_tokens, motif_dict):
    """
    Analyze which motifs are present in a window.
    Returns dict of {category: count}, sorted by count descending.
    """
    counts = count_motifs_in_window(window_tokens, motif_dict)
    
    # Sort by count, descending
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    # Filter to only non-zero
    filtered = {k: v for k, v in sorted_counts.items() if v > 0}
    
    return filtered


def plot_peaks_and_valleys(results_df, tokens, window_size, step_size,
                           motif_dict, output_prefix, n_peaks=3):
    """
    Identify and visualize top peaks and valleys in Sigma with text excerpts.
    """
    sigma_values = results_df['Sigma'].values
    
    # Find peaks (local maxima)
    peak_indices, _ = find_peaks(sigma_values, height=np.percentile(sigma_values, 75))
    peak_values = sigma_values[peak_indices]
    
    # Sort peaks by height
    sorted_peak_idx = np.argsort(peak_values)[::-1]
    top_peaks = [(peak_indices[i], peak_values[i]) for i in sorted_peak_idx[:n_peaks]]
    
    # Find valleys (local minima) - invert signal
    valley_indices, _ = find_peaks(-sigma_values, height=-np.percentile(sigma_values, 25))
    valley_values = sigma_values[valley_indices]
    
    # Sort valleys by depth
    sorted_valley_idx = np.argsort(valley_values)
    top_valleys = [(valley_indices[i], valley_values[i]) for i in sorted_valley_idx[:n_peaks]]
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # Main time series plot (top row, spans all columns)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(results_df['window_index'], sigma_values, 
                color='crimson', linewidth=2, label='Sigma (Σ)')
    ax_main.fill_between(results_df['window_index'], sigma_values, 
                        alpha=0.2, color='crimson')
    
    # Mark peaks
    for idx, val in top_peaks:
        ax_main.scatter(idx, val, color='darkred', s=100, zorder=5, 
                       marker='^', edgecolors='black', linewidths=1.5)
    
    # Mark valleys
    for idx, val in top_valleys:
        ax_main.scatter(idx, val, color='darkblue', s=100, zorder=5, 
                       marker='v', edgecolors='black', linewidths=1.5)
    
    ax_main.set_xlabel('Window Index', fontsize=12)
    ax_main.set_ylabel('Σ (bits/token)', fontsize=12)
    ax_main.set_title('Sigma Peaks (▲) and Valleys (▼) with Text Excerpts', 
                     fontsize=14, fontweight='bold')
    ax_main.grid(alpha=0.3)
    ax_main.legend(loc='upper right')
    
    # Peak excerpts (second row)
    for i, (idx, val) in enumerate(top_peaks):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')
        
        # Extract text
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        excerpt = text_data['window_text'][:300] + '...'
        
        # Wrap text
        import textwrap
        wrapped_excerpt = '\n'.join(textwrap.wrap(excerpt, width=40))
        
        # Get dominant motifs
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        # Format motif info
        motif_text = 'Dominant motifs:\n'
        for cat, count in list(motifs.items())[:3]:
            motif_text += f'• {cat}: {count}\n'
        
        # Title
        title_text = f'Peak #{i+1}\nWindow {idx} | Σ = {val:.4f}'
        
        # Display text
        ax.text(0.5, 0.95, title_text, 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.6))
        
        ax.text(0.5, 0.75, motif_text, 
               transform=ax.transAxes, fontsize=8,
               ha='center', va='top', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.text(0.5, 0.50, wrapped_excerpt, 
               transform=ax.transAxes, fontsize=7,
               ha='center', va='top', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                        alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # Valley excerpts (third row)
    for i, (idx, val) in enumerate(top_valleys):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
        
        # Extract text
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        excerpt = text_data['window_text'][:300] + '...'
        
        # Wrap text
        import textwrap
        wrapped_excerpt = '\n'.join(textwrap.wrap(excerpt, width=40))
        
        # Get dominant motifs
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        # Format motif info
        motif_text = 'Dominant motifs:\n'
        for cat, count in list(motifs.items())[:3]:
            motif_text += f'• {cat}: {count}\n'
        if not motifs:
            motif_text += '(minimal motif presence)'
        
        # Title
        title_text = f'Valley #{i+1}\nWindow {idx} | Σ = {val:.4f}'
        bgcolor = 'lightblue'
        
        # Display text
        ax.text(0.5, 0.95, title_text, 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=bgcolor, alpha=0.6))
        
        ax.text(0.5, 0.75, motif_text, 
               transform=ax.transAxes, fontsize=8,
               ha='center', va='top', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.text(0.5, 0.50, wrapped_excerpt, 
               transform=ax.transAxes, fontsize=7,
               ha='center', va='top', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                        alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    filename = f'{output_prefix}_peaks_valleys.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()
    
    # Export detailed CSV
    export_peaks_valleys_csv(top_peaks, top_valleys, tokens, window_size, step_size, 
                             motif_dict, output_prefix)


def export_peaks_valleys_csv(peaks, valleys, tokens, window_size, step_size,
                             motif_dict, output_prefix):
    """
    Export peak and valley text excerpts to CSV for detailed analysis.
    """
    data = []
    
    # Add peaks
    for rank, (idx, val) in enumerate(peaks, 1):
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        data.append({
            'type': 'Peak',
            'rank': rank,
            'window_idx': idx,
            'sigma_value': val,
            'excerpt': text_data['window_text'][:500],
            'top_motif_1': list(motifs.keys())[0] if len(motifs) > 0 else '',
            'top_motif_1_count': list(motifs.values())[0] if len(motifs) > 0 else 0,
            'top_motif_2': list(motifs.keys())[1] if len(motifs) > 1 else '',
            'top_motif_2_count': list(motifs.values())[1] if len(motifs) > 1 else 0,
            'top_motif_3': list(motifs.keys())[2] if len(motifs) > 2 else '',
            'top_motif_3_count': list(motifs.values())[2] if len(motifs) > 2 else 0,
        })
    
    # Add valleys
    for rank, (idx, val) in enumerate(valleys, 1):
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        data.append({
            'type': 'Valley',
            'rank': rank,
            'window_idx': idx,
            'sigma_value': val,
            'excerpt': text_data['window_text'][:500],
            'top_motif_1': list(motifs.keys())[0] if len(motifs) > 0 else '',
            'top_motif_1_count': list(motifs.values())[0] if len(motifs) > 0 else 0,
            'top_motif_2': list(motifs.keys())[1] if len(motifs) > 1 else '',
            'top_motif_2_count': list(motifs.values())[1] if len(motifs) > 1 else 0,
            'top_motif_3': list(motifs.keys())[2] if len(motifs) > 2 else '',
            'top_motif_3_count': list(motifs.values())[2] if len(motifs) > 2 else 0,
        })
    
    df = pd.DataFrame(data)
    filename = f'{output_prefix}_peaks_valleys_text.csv'
    df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")


# ============================================================================
# PUBLICATION STATISTICS
# ============================================================================

def print_publication_statistics(results_df, text_path, total_tokens, 
                                 window_size, n_windows, n_phrases, is_shuffle=False):
    """
    Print comprehensive statistics for publication.
    """
    label = "SHUFFLE" if is_shuffle else "ORIGINAL"
    print(f"\n{'='*70}")
    print(f"PUBLICATION-READY STATISTICS ({label})")
    print(f"{'='*70}")
    print(f"Text: {os.path.basename(text_path)}")
    print(f"Total semantic tokens: {total_tokens:,}")
    print(f"  (includes {n_phrases} merged multi-word phrases)")
    print(f"Window size: {window_size} tokens")
    print(f"Number of windows: {n_windows}")
    print(f"")
    print(f"Shannon Entropy (H):")
    print(f"  Mean: {results_df['H'].mean():.4f} ± {results_df['H'].std():.4f} bits/token")
    print(f"  Range: [{results_df['H'].min():.4f}, {results_df['H'].max():.4f}]")
    print(f"")
    print(f"Sigma (Σ):")
    print(f"  Mean: {results_df['Sigma'].mean():.6f} ± {results_df['Sigma'].std():.6f} bits/token")
    print(f"  Range: [{results_df['Sigma'].min():.6f}, {results_df['Sigma'].max():.6f}]")
    print(f"")
    print(f"Symbolic Entropy (SE = H + Σ):")
    print(f"  Mean: {results_df['SE'].mean():.4f} ± {results_df['SE'].std():.4f} bits/token")
    print(f"  Range: [{results_df['SE'].min():.4f}, {results_df['SE'].max():.4f}]")
    print(f"")
    print(f"Note: 'bits/token' refers to bits per SEMANTIC TOKEN")
    print(f"      Multi-word phrases are merged into single tokens")
    print(f"{'='*70}\n")


# ============================================================================
# THREE-WAY COMPARISON VISUALIZATION
# ============================================================================

def plot_3way_comparison(orig_kl, word_kl, sent_kl, motif_dict, output_prefix, z_limits):
    """
    Create side-by-side KL heatmaps for all three conditions.
    All use the same z-axis from the original for valid comparison.
    """
    # Create figure with extra space at bottom for colorbar
    fig = plt.figure(figsize=(24, 10))
    
    # Create GridSpec: 3 columns for heatmaps, separate row for colorbar
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], hspace=0.25)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[1, :])
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    
    vmin_k, vmax_k = z_limits['kl']
    
    titles = ['ORIGINAL', 'WORD SHUFFLED', 'SENTENCE SHUFFLED']
    data_sets = [orig_kl, word_kl, sent_kl]
    
    im = None  # Store last valid imshow for colorbar
    
    for ax, title, kl_data in zip(axes, titles, data_sets):
        if kl_data is not None:
            im = ax.imshow(kl_data.T, aspect='auto', cmap='plasma',
                          interpolation='nearest', origin='lower',
                          vmin=vmin_k, vmax=vmax_k)
            ax.set_ylim(-0.5, n_categories - 0.5)
            ax.set_xlabel('Window Index', fontsize=11)
            ax.set_ylabel('Motif Category', fontsize=11)
            ax.set_yticks(range(n_categories))
            ax.set_yticklabels(categories, fontsize=7)  # Reduced from 9 to 7
        else:
            ax.text(0.5, 0.5, 'Not Provided', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add horizontal colorbar in dedicated axes at bottom
    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('KL Contribution (bits/token)', fontsize=12)
    
    fig.suptitle('Three-Way Comparison: Hierarchical Structure Destruction', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    filename = f'{output_prefix}_3way_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def export_validation_summary(orig_df, word_df, sent_df, output_prefix):
    """
    Export comprehensive validation statistics to CSV.
    Includes Cohen's d effect sizes.
    """
    rows = []
    
    # Original stats
    rows.append({
        'Condition': 'Original',
        'H_mean': orig_df['H'].mean(),
        'H_std': orig_df['H'].std(),
        'Sigma_mean': orig_df['Sigma'].mean(),
        'Sigma_std': orig_df['Sigma'].std(),
        'SE_mean': orig_df['SE'].mean(),
        'SE_std': orig_df['SE'].std(),
        'Collapse_Ratio': 1.0,
        'Cohens_d_vs_Original': 0.0
    })
    
    # Word shuffle stats
    if word_df is not None:
        collapse_word = orig_df['Sigma'].mean() / word_df['Sigma'].mean() if word_df['Sigma'].mean() > 0 else float('inf')
        cohens_d_word = calculate_cohens_d(orig_df['Sigma'].values, word_df['Sigma'].values)
        rows.append({
            'Condition': 'Word_Shuffle',
            'H_mean': word_df['H'].mean(),
            'H_std': word_df['H'].std(),
            'Sigma_mean': word_df['Sigma'].mean(),
            'Sigma_std': word_df['Sigma'].std(),
            'SE_mean': word_df['SE'].mean(),
            'SE_std': word_df['SE'].std(),
            'Collapse_Ratio': collapse_word,
            'Cohens_d_vs_Original': cohens_d_word
        })
    
    # Sentence shuffle stats
    if sent_df is not None:
        collapse_sent = orig_df['Sigma'].mean() / sent_df['Sigma'].mean() if sent_df['Sigma'].mean() > 0 else float('inf')
        cohens_d_sent = calculate_cohens_d(orig_df['Sigma'].values, sent_df['Sigma'].values)
        rows.append({
            'Condition': 'Sentence_Shuffle',
            'H_mean': sent_df['H'].mean(),
            'H_std': sent_df['H'].std(),
            'Sigma_mean': sent_df['Sigma'].mean(),
            'Sigma_std': sent_df['Sigma'].std(),
            'SE_mean': sent_df['SE'].mean(),
            'SE_std': sent_df['SE'].std(),
            'Collapse_Ratio': collapse_sent,
            'Cohens_d_vs_Original': cohens_d_sent
        })
    
    df = pd.DataFrame(rows)
    filename = f'{output_prefix}_validation_summary.csv'
    df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Handle command-line arguments
    if len(sys.argv) > 1:
        TEXT_FILE = sys.argv[1]
        print(f"Using command-line specified original file: {TEXT_FILE}")
    if len(sys.argv) > 2:
        WORD_SHUFFLE_FILE = sys.argv[2]
        print(f"Using command-line specified word shuffle file: {WORD_SHUFFLE_FILE}")
    if len(sys.argv) > 3:
        SENT_SHUFFLE_FILE = sys.argv[3]
        print(f"Using command-line specified sentence shuffle file: {SENT_SHUFFLE_FILE}")
    
    # Check original file exists
    if not os.path.exists(TEXT_FILE):
        print(f"ERROR: Original file not found: {TEXT_FILE}")
        print(f"Usage: python {sys.argv[0]} <original> [word_shuffle] [sentence_shuffle]")
        sys.exit(1)
    
    # Generate output prefix from original filename
    output_prefix = os.path.splitext(os.path.basename(TEXT_FILE))[0]
    
    # Storage for comparison
    word_results_df = None
    sent_results_df = None
    word_kl_contributions = None
    sent_kl_contributions = None
    
    # ========================================================================
    # PHASE 1: ANALYZE ORIGINAL TEXT
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: ANALYZING ORIGINAL TEXT")
    print("="*70)
    
    try:
        (results_df, raw_densities, kl_contributions, motif_dict_used, 
         baseline, window_size, total_tokens, tokens) = run_se_analysis(TEXT_FILE, motif_dict)
    except Exception as e:
        print(f"ERROR during original analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    n_windows = len(results_df)
    n_phrases = sum(1 for token in tokens if '_' in token)
    
    # Print statistics for original
    print_publication_statistics(results_df, TEXT_FILE, total_tokens, 
                                 window_size, n_windows, n_phrases, is_shuffle=False)
    
    # Generate visualizations for original
    print("Generating original text visualizations...")
    z_limits = plot_dual_heatmap(results_df, raw_densities, kl_contributions, 
                                 motif_dict_used, output_prefix, z_limits=None, 
                                 condition_label='original')
    
    plot_timeseries(results_df, output_prefix)
    
    step_size = window_size // 2
    plot_peaks_and_valleys(results_df, tokens, window_size, step_size,
                          motif_dict_used, output_prefix)
    
    # Save original results
    csv_filename = f'{output_prefix}_se_results.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Original results saved: {csv_filename}")
    
    # Store original KL for comparison
    orig_kl_contributions = kl_contributions
    
    # ========================================================================
    # PHASE 2: ANALYZE WORD SHUFFLE (if provided)
    # ========================================================================
    if WORD_SHUFFLE_FILE and os.path.exists(WORD_SHUFFLE_FILE):
        print("\n" + "="*70)
        print("PHASE 2: ANALYZING WORD-SHUFFLED TEXT")
        print("="*70)
        
        try:
            (word_results_df, word_raw_densities, word_kl_contributions, 
             _, _, word_window_size, word_total_tokens, word_tokens) = run_se_analysis(WORD_SHUFFLE_FILE, motif_dict)
        except Exception as e:
            print(f"ERROR during word shuffle analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        else:
            word_n_windows = len(word_results_df)
            word_n_phrases = sum(1 for token in word_tokens if '_' in token)
            
            # Print statistics
            print_publication_statistics(word_results_df, WORD_SHUFFLE_FILE, word_total_tokens, 
                                         word_window_size, word_n_windows, word_n_phrases, 
                                         is_shuffle=True)
            
            # Generate heatmap with original z-scale
            print("Generating word shuffle heatmap with original z-axis scale...")
            plot_dual_heatmap(word_results_df, word_raw_densities, word_kl_contributions, 
                             motif_dict_used, output_prefix, z_limits=z_limits,
                             condition_label='word_shuffle')
            
            # Save results
            word_csv = f'{output_prefix}_word_shuffle_se_results.csv'
            word_results_df.to_csv(word_csv, index=False)
            print(f"✓ Word shuffle results saved: {word_csv}")
    
    elif WORD_SHUFFLE_FILE:
        print(f"\n⚠ Warning: Word shuffle file specified but not found: {WORD_SHUFFLE_FILE}")
    
    # ========================================================================
    # PHASE 3: ANALYZE SENTENCE SHUFFLE (if provided)
    # ========================================================================
    if SENT_SHUFFLE_FILE and os.path.exists(SENT_SHUFFLE_FILE):
        print("\n" + "="*70)
        print("PHASE 3: ANALYZING SENTENCE-SHUFFLED TEXT")
        print("="*70)
        
        try:
            (sent_results_df, sent_raw_densities, sent_kl_contributions, 
             _, _, sent_window_size, sent_total_tokens, sent_tokens) = run_se_analysis(SENT_SHUFFLE_FILE, motif_dict)
        except Exception as e:
            print(f"ERROR during sentence shuffle analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        else:
            sent_n_windows = len(sent_results_df)
            sent_n_phrases = sum(1 for token in sent_tokens if '_' in token)
            
            # Print statistics
            print_publication_statistics(sent_results_df, SENT_SHUFFLE_FILE, sent_total_tokens, 
                                         sent_window_size, sent_n_windows, sent_n_phrases, 
                                         is_shuffle=True)
            
            # Generate heatmap with original z-scale
            print("Generating sentence shuffle heatmap with original z-axis scale...")
            plot_dual_heatmap(sent_results_df, sent_raw_densities, sent_kl_contributions, 
                             motif_dict_used, output_prefix, z_limits=z_limits,
                             condition_label='sent_shuffle')
            
            # Save results
            sent_csv = f'{output_prefix}_sent_shuffle_se_results.csv'
            sent_results_df.to_csv(sent_csv, index=False)
            print(f"✓ Sentence shuffle results saved: {sent_csv}")
    
    elif SENT_SHUFFLE_FILE:
        print(f"\n⚠ Warning: Sentence shuffle file specified but not found: {SENT_SHUFFLE_FILE}")
    
    # ========================================================================
    # PHASE 4: THREE-WAY COMPARISON
    # ========================================================================
    if word_kl_contributions is not None or sent_kl_contributions is not None:
        print("\n" + "="*70)
        print("PHASE 4: THREE-WAY COMPARISON")
        print("="*70)
        
        # Generate side-by-side comparison
        print("Generating three-way comparison visualization...")
        plot_3way_comparison(orig_kl_contributions, word_kl_contributions, 
                            sent_kl_contributions, motif_dict_used, output_prefix, z_limits)
        
        # Export validation summary with Cohen's d
        print("Exporting validation summary...")
        summary_df = export_validation_summary(results_df, word_results_df, 
                                               sent_results_df, output_prefix)
        
        # Print validation results
        print("\n" + "="*70)
        print("THREE-WAY VALIDATION RESULTS")
        print("="*70)
        
        original_sigma_mean = results_df['Sigma'].mean()
        print(f"Original Σ (mean): {original_sigma_mean:.6f} bits/token")
        print()
        
        if word_results_df is not None:
            word_sigma_mean = word_results_df['Sigma'].mean()
            word_collapse = original_sigma_mean / word_sigma_mean if word_sigma_mean > 0 else float('inf')
            word_cohens_d = calculate_cohens_d(results_df['Sigma'].values, word_results_df['Sigma'].values)
            print(f"Word Shuffle Σ (mean):  {word_sigma_mean:.6f} bits/token")
            print(f"  → Collapse ratio: {word_collapse:.2f}x")
            print(f"  → Cohen's d: {word_cohens_d:.2f}")
            if word_collapse > 10:
                print("  ✓ STRONG: Σ collapsed >10x")
            elif word_collapse > 5:
                print("  ⚠ MODERATE: Σ collapsed 5-10x")
            else:
                print("  ✗ WEAK: Σ collapsed <5x")
            print()
        
        if sent_results_df is not None:
            sent_sigma_mean = sent_results_df['Sigma'].mean()
            sent_collapse = original_sigma_mean / sent_sigma_mean if sent_sigma_mean > 0 else float('inf')
            sent_cohens_d = calculate_cohens_d(results_df['Sigma'].values, sent_results_df['Sigma'].values)
            print(f"Sentence Shuffle Σ (mean): {sent_sigma_mean:.6f} bits/token")
            print(f"  → Collapse ratio: {sent_collapse:.2f}x")
            print(f"  → Cohen's d: {sent_cohens_d:.2f}")
            if sent_collapse > 3:
                print("  ✓ Expected intermediate collapse")
            else:
                print("  ⚠ Lower than expected (sentence structure may preserve some clustering)")
            print()
        
        # Hierarchical validation check
        if word_results_df is not None and sent_results_df is not None:
            word_sigma = word_results_df['Sigma'].mean()
            sent_sigma = sent_results_df['Sigma'].mean()
            if word_sigma < sent_sigma < original_sigma_mean:
                print("✓ HIERARCHICAL VALIDATION PASSED:")
                print(f"  Original ({original_sigma_mean:.6f}) > Sentence ({sent_sigma:.6f}) > Word ({word_sigma:.6f})")
                print("  This confirms SE detects multiple levels of semantic structure.")
            else:
                print("⚠ Hierarchical ordering not as expected.")
                print(f"  Original: {original_sigma_mean:.6f}, Sentence: {sent_sigma:.6f}, Word: {word_sigma:.6f}")
        
        print("="*70)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Original text outputs:")
    print(f"  - {output_prefix}_se_heatmap.png")
    print(f"  - {output_prefix}_se_timeseries.png")
    print(f"  - {output_prefix}_peaks_valleys.png")
    print(f"  - {output_prefix}_se_results.csv")
    print(f"  - {output_prefix}_peaks_valleys_text.csv")
    
    if word_results_df is not None:
        print(f"\nWord shuffle outputs:")
        print(f"  - {output_prefix}_word_shuffle_heatmap.png")
        print(f"  - {output_prefix}_word_shuffle_se_results.csv")
    
    if sent_results_df is not None:
        print(f"\nSentence shuffle outputs:")
        print(f"  - {output_prefix}_sent_shuffle_heatmap.png")
        print(f"  - {output_prefix}_sent_shuffle_se_results.csv")
    
    if word_kl_contributions is not None or sent_kl_contributions is not None:
        print(f"\nComparison outputs:")
        print(f"  - {output_prefix}_3way_comparison.png")
        print(f"  - {output_prefix}_validation_summary.csv")
    
    print(f"\nFor publication, cite:")
    print(f"  Kurian, M. (2025). Symbolic Entropy: A Mathematical Framework")
    print(f"  for Quantifying Meaning Density in Text.")
    print(f"{'='*70}\n")
