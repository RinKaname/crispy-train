# Model Comparison and Research

## Baseline Model Analysis
The original `solution.py` implementation used:
- A basic `TfidfVectorizer` for words only.
- A `LogisticRegression` model with balanced class weights.
- It yielded a local CV Macro-F1 score of approximately `0.635`.

## Feature Engineering Enhancements
1. **Adding Contextual Columns:** I initially tried appending the 'county' and 'year' to the textual features, but the main driver of information is the language itself.
2. **Character N-grams:** Since trades are described in 18th-century spelling with abbreviations, using word boundaries may miss morphological similarities. By utilizing a `FeatureUnion` of Word N-grams (1-3) and Character N-grams (2-5, within word boundaries), the model captures sub-word patterns and root words effectively.
    - Result: This significantly improved the local CV Macro-F1 score to **`0.697`**.

## Model Architectures
Evaluated multiple models on the unified features:
- **Logistic Regression (L2, C=2.0)**: CV Macro-F1 = `0.697`
- **LinearSVC (C=1.0)**: CV Macro-F1 = `0.688`
- **SGDClassifier (Log Loss)**: CV Macro-F1 = `0.690`
- **RandomForestClassifier**: CV Macro-F1 = `0.618`

**Conclusion**: Logistic Regression remained the top choice due to its high performance and its ability to output calibrated probabilities, which is essential for self-training.

## Unlabelled Data Utilization (Semi-Supervised Learning)
The competition prompt heavily emphasizes that the training and test distributions have different priors, and the unlabelled data should be used.
- **Pseudo-labeling / Self-Training**: I implemented transductive self-training, evaluating high-confidence predictions on the combined pool of unlabelled data and test data.
- By using an initial Logistic Regression model (without balanced class weights) to generate pseudo-labels for the unlabelled pool and test set, we avoid overfitting to the skewed training prior.
- High-confidence predictions (probability > 0.8) are iteratively added to the training set.
- A final model is then trained on the expanded dataset *with* balanced class weights.
- While the local CV score on the training set distribution drops slightly (to `~0.688`), this is expected because the unlabelled data has a different distribution. In the final `solution.py`, this approach helps the model adapt to the test distribution.

## Final Plan for `solution.py`
The final script will execute:
1. Concatenation of transcription and commodity text.
2. Vectorization via `FeatureUnion` combining word (1-3) and char (2-5) n-grams.
3. Iterative Transductive Self-Training over the unlabelled and test datasets.
4. Final prediction using Logistic Regression with balanced weights.
