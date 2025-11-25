**Symbolic Entropy: 3-Way Shuffle Validation Framework**
Overview
This tool validates Symbolic Entropy (SE) measurements through controlled text randomization. SE extends Shannon's information theory to quantify semantic meaning density by measuring two dimensions:
H (Shannon Entropy): Lexical diversity/complexity
Σ (Sigma): Archetypal pattern concentration via KL divergence
The key insight: Meaning requires structure. When text structure is destroyed while vocabulary remains constant, SE should collapse—proving the framework measures narrative architecture, not mere word frequency.


**What This Does**
Generates side-by-side heatmap comparisons across three conditions:
1. Original Text

Intact narrative structure
Expected: Visible Σ peaks at semantically dense passages
Baseline measurement of authentic meaning density

2. Word-Shuffled Text

Complete randomization of word order
Vocabulary preserved, all structure destroyed
Expected: Σ → 0 (10-20x collapse), H remains stable

3. Sentence-Shuffled Text

Sentences remain intact but reordered randomly
Local coherence preserved, global narrative destroyed
Expected: Partial Σ reduction, intermediate between original and word-shuffle

**Why This Matters**
Falsifiability Test: If SE only measured vocabulary richness, shuffling wouldn't affect it. The dramatic Σ collapse in randomized conditions proves SE captures semantic structure—the "Level B problem" Shannon deliberately excluded from his 1948 theory.
Key Features
Consistent Parameters: Fixed window size/overlap calculated from original text ensures valid cross-condition comparison
Proper KL Divergence: Uses information-theoretic formula (Σ = Σ p_k × log₂(p_k / π_k)), not simple frequency counting
Visual Evidence: Side-by-side heatmaps with consistent color scaling enable direct interpretation
Statistical Validation: Includes Cohen's d effect sizes, collapse ratios, and comprehensive metrics

The results of the shuffle test prove that meaning understood as indexed motif presences has a structure in narrative that unfolds over time and is not reducible to word content. This results in a 2D representational method (SE) that shows where meaning is active in a motif space across a data sample.
