import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
import scipy.sparse as sp

def solve():
    """
    Main execution pipeline for inferring occupations from historical billed goods.
    Designed to be executed as: python3 solution.py <public_dir> <submission_out>
    """
    if len(sys.argv) != 3:
        print("Usage: python3 solution.py <public_dir> <submission_out>")
        sys.exit(1)

    public_dir = sys.argv[1]
    submission_out = sys.argv[2]

    print("Initializing data ingestion protocols...")
    entries_path = os.path.join(public_dir, 'entries.csv')
    train_path = os.path.join(public_dir, 'train.csv')
    test_path = os.path.join(public_dir, 'test_suppliers.csv')

    entries_df = pd.read_csv(entries_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Textual Normalization and Feature Aggregation
    print("Normalizing textual matrices...")
    entries_df['transcription'] = entries_df['transcription'].fillna('')
    entries_df['commodities_text'] = entries_df['commodities_text'].fillna('')

    # Synthesize a unified text field
    entries_df['combined_text'] = entries_df['transcription'] + " " + entries_df['commodities_text']

    print("Aggregating dimensional features per supplier_id...")
    # Condense document lines into singular supplier profiles
    supplier_docs = entries_df.groupby('supplier_id')['combined_text'].apply(lambda x: ' '.join(x)).reset_index()
    supplier_docs.columns = ['supplier_id', 'text']

    # 2. Stratify the Labelled, Unlabelled, and Test Sets
    train_data = pd.merge(train_df, supplier_docs, on='supplier_id', how='left')
    test_data = pd.merge(test_df, supplier_docs, on='supplier_id', how='left')

    # Secure against absent textual records
    train_data['text'] = train_data['text'].fillna('')
    test_data['text'] = test_data['text'].fillna('')

    unlabelled_suppliers = set(supplier_docs['supplier_id']) - set(train_df['supplier_id']) - set(test_df['supplier_id'])
    unlabelled_data = supplier_docs[supplier_docs['supplier_id'].isin(unlabelled_suppliers)]

    # 3. Lexical Vectorization Exploiting the Global Corpus
    print("Vectorizing textual parameters across the global pool...")
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=25000,
        sublinear_tf=True,
        stop_words='english'
    )

    char_vectorizer = TfidfVectorizer(
        ngram_range=(2, 5),
        analyzer='char_wb',
        max_features=50000,
        sublinear_tf=True
    )

    vectorizer = FeatureUnion([('word', word_vectorizer), ('char', char_vectorizer)])
    vectorizer.fit(supplier_docs['text'])

    X_train = vectorizer.transform(train_data['text'])
    y_train = train_data['occupation_role']

    X_test = vectorizer.transform(test_data['text'])
    X_unlabelled = vectorizer.transform(unlabelled_data['text'])

    # 4. Transductive Self-Training
    print("Executing Transductive Self-Training...")

    # Combine unlabelled and test sets for the unlabelled pool
    X_curr_unlabelled = sp.vstack([X_unlabelled, X_test]).copy()

    X_train_expanded = X_train.copy()
    y_train_expanded = y_train.copy()

    # Self-training iterations
    iterations = 3
    for iteration in range(iterations):
        print(f"  Iteration {iteration + 1}/{iterations}...")

        # Fit model without balanced weights to get probability distributions matching the pool
        model_iter = LogisticRegression(class_weight=None, max_iter=1500, C=2.0, random_state=42, solver='lbfgs')
        model_iter.fit(X_train_expanded, y_train_expanded)

        if X_curr_unlabelled.shape[0] == 0:
            break

        unlabelled_probs = model_iter.predict_proba(X_curr_unlabelled)
        unlabelled_preds = model_iter.classes_[np.argmax(unlabelled_probs, axis=1)]
        unlabelled_max_probs = np.max(unlabelled_probs, axis=1)

        # High confidence threshold
        threshold = 0.8
        high_conf_indices = np.where(unlabelled_max_probs >= threshold)[0]

        if len(high_conf_indices) == 0:
            print("    No high confidence predictions found. Terminating early.")
            break

        print(f"    Adding {len(high_conf_indices)} high confidence pseudo-labels...")

        X_pseudo = X_curr_unlabelled[high_conf_indices]
        y_pseudo = unlabelled_preds[high_conf_indices]

        X_train_expanded = sp.vstack([X_train_expanded, X_pseudo])
        y_train_expanded = pd.concat([y_train_expanded, pd.Series(y_pseudo)])

        # Remove added pseudo-labels from unlabelled pool
        mask = np.ones(X_curr_unlabelled.shape[0], dtype=bool)
        mask[high_conf_indices] = False
        X_curr_unlabelled = X_curr_unlabelled[mask]

    print(f"Final training set size: {X_train_expanded.shape[0]} (Original: {X_train.shape[0]})")

    # 5. Final Model Architecture and Training
    print("Executing final optimization mapping (Logistic Regression)...")
    final_model = LogisticRegression(
        class_weight='balanced', # Crucial mechanism to optimize macro-F1 score
        max_iter=1500,
        C=2.0,
        random_state=42,
        solver='lbfgs'
    )
    final_model.fit(X_train_expanded, y_train_expanded)

    # 6. Inference and Formatting
    print("Generating predictive inferences for the test distribution...")
    test_predictions = final_model.predict(X_test)

    submission_df = pd.DataFrame({
        'supplier_id': test_data['supplier_id'],
        'prediction': test_predictions
    })

    print(f"Exporting final matrices to {submission_out}...")
    os.makedirs(os.path.dirname(submission_out), exist_ok=True)
    submission_df.to_csv(submission_out, index=False)

    print("Execution sequence successfully terminated.")

if __name__ == "__main__":
    solve()
