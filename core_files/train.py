#!/usr/bin/env python3
"""
Training script for EdTech Feedback Validation Model
"""

import os
import sys
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess_data
from model import FeedbackValidationModel

def train_model(train_path: str = 'train.xlsx', 
                eval_path: str = 'evaluation.xlsx',
                model_save_path: str = 'feedback_validation_model.pth',
                epochs: int = 3,
                batch_size: int = 16,
                learning_rate: float = 2e-5,
                max_length: int = 128,
                model_name: str = 'bert-base-uncased'):
    """
    Train the feedback validation model
    """
    print("🚀 Starting EdTech Feedback Validation Model Training")
    print("=" * 60)
    
    # Check if data files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation data file not found: {eval_path}")
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    train_df, eval_df = load_and_preprocess_data(train_path, eval_path)
    
    # Initialize model
    print(f"🤖 Initializing {model_name} model...")
    model = FeedbackValidationModel(model_name=model_name, max_length=max_length)
    
    # Create datasets
    print("📝 Creating training and evaluation datasets...")
    train_loader, eval_loader = model.create_datasets(train_df, eval_df, batch_size=batch_size)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Evaluation samples: {len(eval_loader.dataset)}")
    
    # Train model
    print(f"🎯 Training model for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Evaluation samples: {len(eval_loader.dataset)}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total training steps: {len(train_loader) * epochs}")
    print("-" * 50)
    
    start_time = datetime.now()
    print("🚀 Starting training process...")
    print("💡 You'll see progress bars and real-time metrics below:")
    print("=" * 50)
    
    history = model.train(
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )
    end_time = datetime.now()
    
    training_time = end_time - start_time
    print("=" * 50)
    print(f"⏱️  Training completed in: {training_time}")
    print(f"📊 Average time per epoch: {training_time / epochs}")
    
    # Generate evaluation report
    print("📈 Generating evaluation report...")
    results = model.generate_evaluation_report(eval_loader, 'evaluation_report.png')
    
    # Save model
    print(f"💾 Saving model to {model_save_path}...")
    model.save_model(model_save_path)
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Final Results:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1-Score: {results['f1_score']:.4f}")
    print(f"   Model saved to: {model_save_path}")
    print(f"   Evaluation report saved to: evaluation_report.png")
    
    # Plot training history
    plot_training_history(history)
    
    return model, results

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Evaluation loss
    axes[0, 1].plot(history['eval_loss'], label='Evaluation Loss', color='red')
    axes[0, 1].set_title('Evaluation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Evaluation accuracy
    axes[1, 0].plot(history['eval_accuracy'], label='Evaluation Accuracy', color='green')
    axes[1, 0].set_title('Evaluation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Evaluation F1-score
    axes[1, 1].plot(history['eval_f1'], label='Evaluation F1-Score', color='orange')
    axes[1, 1].set_title('Evaluation F1-Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("📊 Training history plot saved to: training_history.png")
    plt.show()

def main():
    """
    Main function with command line argument parsing
    """
    parser = argparse.ArgumentParser(description='Train EdTech Feedback Validation Model')
    
    parser.add_argument('--train_path', type=str, default='train.xlsx',
                       help='Path to training data file (default: train.xlsx)')
    parser.add_argument('--eval_path', type=str, default='evaluation.xlsx',
                       help='Path to evaluation data file (default: evaluation.xlsx)')
    parser.add_argument('--model_save_path', type=str, default='feedback_validation_model.pth',
                       help='Path to save the trained model (default: feedback_validation_model.pth)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pre-trained model name (default: bert-base-uncased)')
    
    args = parser.parse_args()
    
    try:
        # Train the model
        model, results = train_model(
            train_path=args.train_path,
            eval_path=args.eval_path,
            model_save_path=args.model_save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            model_name=args.model_name
        )
        
        print("\n✅ Training completed successfully!")
        print("🎯 You can now run the Gradio app with: python app.py")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

