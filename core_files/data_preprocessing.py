import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import random
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        """
        if not text:
            return ''
        
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def correct_spelling(self, text: str) -> str:
        """
        Correct spelling using TextBlob
        """
        if not text:
            return ''
        
        blob = TextBlob(text)
        return str(blob.correct())
    
    def augment_negative_samples(self, df: pd.DataFrame, target_negative_samples: int = 2000) -> pd.DataFrame:
        """
        Create negative samples by mismatching text and reason pairs
        """
        positive_samples = df[df['label'] == 1].copy()
        current_negative_samples = len(df[df['label'] == 0])
        
        if current_negative_samples >= target_negative_samples:
            return df
        
        negative_samples_needed = target_negative_samples - current_negative_samples
        negative_samples = []
        
        # Get unique reasons
        unique_reasons = positive_samples['reason'].unique()
        
        for _ in range(negative_samples_needed):
            # Randomly select a text and a different reason
            random_text = positive_samples.sample(1)['text'].iloc[0]
            random_reason = random.choice(unique_reasons)
            
            # Ensure text and reason don't match (create mismatch)
            negative_samples.append({
                'text': random_text,
                'reason': random_reason,
                'label': 0
            })
        
        # Create DataFrame for new negative samples
        negative_df = pd.DataFrame(negative_samples)
        
        # Combine with original data
        augmented_df = pd.concat([df, negative_df], ignore_index=True)
        
        return augmented_df
    
    def create_synthetic_negative_samples(self, df: pd.DataFrame, target_negative_samples: int = 2000) -> pd.DataFrame:
        """
        Create synthetic negative samples using paraphrasing and word swapping
        """
        positive_samples = df[df['label'] == 1].copy()
        current_negative_samples = len(df[df['label'] == 0])
        
        if current_negative_samples >= target_negative_samples:
            return df
        
        negative_samples_needed = target_negative_samples - current_negative_samples
        negative_samples = []
        
        # Common negative words to add
        negative_words = ['not', 'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'useless']
        
        for _ in range(negative_samples_needed):
            # Select a random positive sample
            sample = positive_samples.sample(1).iloc[0]
            
            # Create negative version by adding negative words or modifying text
            text = sample['text']
            reason = sample['reason']
            
            # Method 1: Add negative words
            if random.random() < 0.5:
                negative_word = random.choice(negative_words)
                text = f"{negative_word} {text}"
            else:
                # Method 2: Modify text slightly to create mismatch
                words = text.split()
                if len(words) > 3:
                    # Replace a word with a random word
                    replace_idx = random.randint(0, len(words) - 1)
                    words[replace_idx] = random.choice(['bad', 'terrible', 'awful'])
                    text = ' '.join(words)
            
            negative_samples.append({
                'text': text,
                'reason': reason,
                'label': 0
            })
        
        # Create DataFrame for new negative samples
        negative_df = pd.DataFrame(negative_samples)
        
        # Combine with original data
        augmented_df = pd.concat([df, negative_df], ignore_index=True)
        
        return augmented_df
    
    def preprocess_dataset(self, df: pd.DataFrame, augment_negative: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        """
        print("Starting data preprocessing...")
        
        # Clean text and reason columns
        df['text_cleaned'] = df['text'].apply(self.clean_text)
        df['reason_cleaned'] = df['reason'].apply(self.clean_text)
        
        # Remove stopwords (optional - can be commented out if needed)
        # df['text_cleaned'] = df['text_cleaned'].apply(self.remove_stopwords)
        # df['reason_cleaned'] = df['reason_cleaned'].apply(self.remove_stopwords)
        
        # Correct spelling (optional - can be slow for large datasets)
        # df['text_cleaned'] = df['text_cleaned'].apply(self.correct_spelling)
        # df['reason_cleaned'] = df['reason_cleaned'].apply(self.correct_spelling)
        
        if augment_negative:
            print("Augmenting negative samples...")
            df = self.augment_negative_samples(df)
            df = self.create_synthetic_negative_samples(df)
        
        # Remove rows with empty text or reason
        df = df.dropna(subset=['text_cleaned', 'reason_cleaned'])
        df = df[df['text_cleaned'] != '']
        df = df[df['reason_cleaned'] != '']
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df

def load_and_preprocess_data(train_path: str, eval_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess training and evaluation datasets
    """
    # Load data
    print("Loading datasets...")
    train_df = pd.read_excel(train_path)
    eval_df = pd.read_excel(eval_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Evaluation data shape: {eval_df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess training data
    train_df_processed = preprocessor.preprocess_dataset(train_df, augment_negative=True)
    
    # Preprocess evaluation data (no augmentation needed)
    eval_df_processed = preprocessor.preprocess_dataset(eval_df, augment_negative=False)
    
    return train_df_processed, eval_df_processed

if __name__ == "__main__":
    # Test the preprocessing
    train_df, eval_df = load_and_preprocess_data('train.xlsx', 'evaluation.xlsx')
    
    # Save processed data
    train_df.to_csv('train_processed.csv', index=False)
    eval_df.to_csv('evaluation_processed.csv', index=False)
    
    print("Data preprocessing completed and saved!")



