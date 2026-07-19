import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Enable tqdm for pandas operations to visualize processing duration in the logs
tqdm.pandas()

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
    
    # Synthesize a unified text field to maximize context extraction
    entries_df['combined_text'] = entries_df['transcription'] + " " + entries_df['commodities_text']
    
    print("Aggregating dimensional features per supplier_id...")
    # Condense document lines into singular supplier profiles using tqdm
    supplier_docs = entries_df.groupby('supplier_id')['combined_text'].progress_apply(lambda x: ' '.join(x)).reset_index()
    supplier_docs.columns = ['supplier_id', 'text']
    
    # 2. Stratify the Labelled and Unlabelled Sets
    train_data = pd.merge(train_df, supplier_docs, on='supplier_id', how='left')
    test_data = pd.merge(test_df, supplier_docs, on='supplier_id', how='left')
    
    # Secure against absent textual records in the test subset
    test_data['text'] = test_data['text'].fillna('') 
    
    # 3. Lexical Vectorization Exploiting the Global Corpus
    print("Vectorizing textual parameters across the unlabelled and labelled pools...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), 
        max_features=60000, 
        sublinear_tf=True, 
        stop_words='english'
    )
    
    # Fitting on the ENTIRE supplier space (train + test + unlabelled) to construct a resilient vocabulary
    vectorizer.fit(supplier_docs['text'])
    
    X_train = vectorizer.transform(train_data['text'])
    y_train = train_data['occupation_role']
    
    X_test = vectorizer.transform(test_data['text'])
    
    # 4. Model Architecture and Training
    print("Executing optimization mapping (Logistic Regression)...")
    model = LogisticRegression(
        class_weight='balanced', # Crucial mechanism to optimize macro-F1 score
        max_iter=1500, 
        C=2.0,
        random_state=42,
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    
    # 5. Inference and Formatting
    print("Generating predictive inferences for the test distribution...")
    test_predictions = model.predict(X_test)
    
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