import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

class FeedbackDataset(Dataset):
    def __init__(self, texts: List[str], reasons: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.reasons = reasons
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        reason = str(self.reasons[idx])
        label = int(self.labels[idx])
        
        # Combine text and reason with separator
        combined_text = f"{text} [SEP] {reason}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class FeedbackClassifier(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2):
        super(FeedbackClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class FeedbackValidationModel:
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def create_datasets(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, 
                       batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and evaluation datasets
        """
        # Use cleaned text if available, otherwise use original
        train_texts = train_df.get('text_cleaned', train_df['text']).tolist()
        train_reasons = train_df.get('reason_cleaned', train_df['reason']).tolist()
        train_labels = train_df['label'].tolist()
        
        eval_texts = eval_df.get('text_cleaned', eval_df['text']).tolist()
        eval_reasons = eval_df.get('reason_cleaned', eval_df['reason']).tolist()
        eval_labels = eval_df['label'].tolist()
        
        # Create datasets
        train_dataset = FeedbackDataset(train_texts, train_reasons, train_labels, 
                                      self.tokenizer, self.max_length)
        eval_dataset = FeedbackDataset(eval_texts, eval_reasons, eval_labels, 
                                     self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, eval_loader
    
    def train(self, train_loader: DataLoader, eval_loader: DataLoader, 
              epochs: int = 3, learning_rate: float = 2e-5, 
              weight_decay: float = 0.01) -> Dict[str, List[float]]:
        """
        Train the model
        """
        # Initialize model
        self.model = FeedbackClassifier(self.model_name).to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': []
        }
        
        print("Starting training...")
        print(f"Total training steps: {len(train_loader) * epochs}")
        print(f"Training on device: {self.device}")
        print("-" * 50)
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", 
                            unit="batch", ncols=100)
            
            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                
                # Update progress bar
                avg_loss_so_far = total_train_loss / (batch_idx + 1)
                train_pbar.set_postfix({
                    'Loss': f'{avg_loss_so_far:.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Evaluation phase
            print(f"📊 Evaluating epoch {epoch + 1}...")
            eval_loss, eval_accuracy, eval_f1 = self.evaluate(eval_loader, criterion)
            history['eval_loss'].append(eval_loss)
            history['eval_accuracy'].append(eval_accuracy)
            history['eval_f1'].append(eval_f1)
            
            epoch_time = time.time() - epoch_start_time
            print(f"⏱️  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
            print(f"📈 Train Loss: {avg_train_loss:.4f}")
            print(f"📊 Eval Loss: {eval_loss:.4f}")
            print(f"🎯 Eval Accuracy: {eval_accuracy:.4f}")
            print(f"📊 Eval F1-Score: {eval_f1:.4f}")
            print("-" * 50)
        
        return history
    
    def evaluate(self, eval_loader: DataLoader, criterion=None) -> Tuple[float, float, float]:
        """
        Evaluate the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        total_eval_loss = 0
        predictions = []
        true_labels = []
        
        # Progress bar for evaluation
        eval_pbar = tqdm(eval_loader, desc="Evaluating", unit="batch", ncols=80)
        
        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if criterion:
                    loss = criterion(outputs, labels)
                    total_eval_loss += loss.item()
                
                logits = outputs.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(label_ids)
        
        avg_eval_loss = total_eval_loss / len(eval_loader) if criterion else 0
        accuracy = accuracy_score(true_labels, predictions)
        _, _, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        
        return avg_eval_loss, accuracy, f1
    
    def predict(self, text: str, reason: str) -> Tuple[int, float]:
        """
        Make prediction for a single text-reason pair
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Combine text and reason
        combined_text = f"{text} [SEP] {reason}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        return prediction, confidence
    
    def save_model(self, path: str):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_name': self.model_name,
            'max_length': self.max_length
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained model
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model = FeedbackClassifier(checkpoint['model_name']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        self.max_length = checkpoint['max_length']
        print(f"Model loaded from {path}")
    
    def generate_evaluation_report(self, eval_loader: DataLoader, save_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(label_ids)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Confusion matrix heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Aligned', 'Aligned'],
                   yticklabels=['Not Aligned', 'Aligned'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Metrics bar chart
        plt.subplot(2, 2, 2)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        bars = plt.bar(metrics, values, color=colors)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Training history (if available)
        plt.subplot(2, 2, 3)
        plt.text(0.1, 0.5, f'Final Results:\n\nAccuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}', 
                fontsize=12, verticalalignment='center')
        plt.axis('off')
        
        # Classification report
        plt.subplot(2, 2, 4)
        report = classification_report(true_labels, predictions, 
                                     target_names=['Not Aligned', 'Aligned'], 
                                     output_dict=True)
        report_text = f"Classification Report:\n\n"
        report_text += f"Not Aligned:\n"
        report_text += f"  Precision: {report['Not Aligned']['precision']:.3f}\n"
        report_text += f"  Recall: {report['Not Aligned']['recall']:.3f}\n"
        report_text += f"  F1-Score: {report['Not Aligned']['f1-score']:.3f}\n\n"
        report_text += f"Aligned:\n"
        report_text += f"  Precision: {report['Aligned']['precision']:.3f}\n"
        report_text += f"  Recall: {report['Aligned']['recall']:.3f}\n"
        report_text += f"  F1-Score: {report['Aligned']['f1-score']:.3f}"
        
        plt.text(0.1, 0.5, report_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation report saved to {save_path}")
        
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }

if __name__ == "__main__":
    # Test the model
    from data_preprocessing import load_and_preprocess_data
    
    # Load and preprocess data
    train_df, eval_df = load_and_preprocess_data('train.xlsx', 'evaluation.xlsx')
    
    # Initialize model
    model = FeedbackValidationModel()
    
    # Create datasets
    train_loader, eval_loader = model.create_datasets(train_df, eval_df)
    
    # Train model
    history = model.train(train_loader, eval_loader, epochs=2)
    
    # Generate evaluation report
    results = model.generate_evaluation_report(eval_loader, 'evaluation_report.png')
    
    # Save model
    model.save_model('feedback_validation_model.pth')
    
    print("Training completed!")
