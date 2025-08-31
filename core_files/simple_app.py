#!/usr/bin/env python3
"""
Simple command-line interface for EdTech Feedback Validation System
"""

import torch
import pandas as pd
import numpy as np
from model import FeedbackValidationModel
from data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class SimpleFeedbackValidationApp:
    def __init__(self, model_path: str = None):
        self.model = FeedbackValidationModel()
        self.preprocessor = DataPreprocessor()
        
        # Load model if path is provided
        if model_path:
            try:
                self.model.load_model(model_path)
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("Training new model...")
                self.train_model()
        else:
            print("No model path provided. Training new model...")
            self.train_model()
    
    def train_model(self):
        """Train the model if not already trained"""
        try:
            from data_preprocessing import load_and_preprocess_data
            
            print("Loading and preprocessing data...")
            train_df, eval_df = load_and_preprocess_data('train.xlsx', 'evaluation.xlsx')
            
            print("Creating datasets...")
            train_loader, eval_loader = self.model.create_datasets(train_df, eval_df, batch_size=8)
            
            print("Training model...")
            history = self.model.train(train_loader, eval_loader, epochs=2, learning_rate=2e-5)
            
            print("Saving model...")
            self.model.save_model('feedback_validation_model.pth')
            
            print("Model training completed!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise e
    
    def predict_feedback(self, text: str, reason: str) -> tuple:
        """
        Predict whether feedback aligns with reason
        """
        if not text or not reason:
            return "Please provide both text and reason.", 0.0, "Invalid input"
        
        try:
            # Clean the input text
            cleaned_text = self.preprocessor.clean_text(text)
            cleaned_reason = self.preprocessor.clean_text(reason)
            
            # Make prediction
            prediction, confidence = self.model.predict(cleaned_text, cleaned_reason)
            
            # Format output
            result = "✅ ALIGNED" if prediction == 1 else "❌ NOT ALIGNED"
            confidence_pct = confidence * 100
            
            # Create explanation
            if prediction == 1:
                explanation = f"The feedback text aligns well with the selected reason. The model is {confidence_pct:.1f}% confident in this prediction."
            else:
                explanation = f"The feedback text does not align with the selected reason. The model is {confidence_pct:.1f}% confident in this prediction."
            
            return result, confidence_pct, explanation
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0, "An error occurred during prediction"
    
    def batch_predict(self, file_path: str) -> str:
        """
        Process batch predictions from CSV or Excel file
        """
        try:
            # Determine file type and read accordingly
            if file_path.lower().endswith('.csv'):
                # Try different encodings for CSV
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin-1')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='cp1252')
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return "Unsupported file format. Please use CSV (.csv) or Excel (.xlsx, .xls) files."
            
            if 'text' not in df.columns or 'reason' not in df.columns:
                return "File must contain 'text' and 'reason' columns."
            
            results = []
            print(f"Processing {len(df)} samples...")
            
            for idx, row in df.iterrows():
                text = str(row['text'])
                reason = str(row['reason'])
                
                prediction, confidence, _ = self.predict_feedback(text, reason)
                results.append({
                    'text': text,
                    'reason': reason,
                    'prediction': prediction,
                    'confidence': confidence
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} samples...")
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results
            output_path = 'batch_predictions.csv'
            results_df.to_csv(output_path, index=False)
            
            # Create summary
            aligned_count = len(results_df[results_df['prediction'].str.contains('ALIGNED')])
            total_count = len(results_df)
            avg_confidence = results_df['confidence'].mean()
            
            summary = f"""
📊 Batch Prediction Results:

- Total samples: {total_count}
- Aligned feedback: {aligned_count}
- Not aligned feedback: {total_count - aligned_count}
- Average confidence: {avg_confidence:.1f}%

Results saved to: {output_path}
            """
            
            return summary
            
        except Exception as e:
            return f"Error processing batch file: {str(e)}"
    
    def interactive_mode(self):
        """Run interactive mode for single predictions"""
        print("\n🎓 EdTech Feedback Validation System - Interactive Mode")
        print("=" * 60)
        print("Enter 'quit' to exit, 'help' for examples")
        
        while True:
            print("\n" + "-" * 40)
            
            # Get user input
            text = input("📝 Enter user feedback text: ").strip()
            
            if text.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif text.lower() == 'help':
                print("\n💡 Example inputs:")
                print("Text: this is an amazing app for online classes!")
                print("Reason: good app for conducting online classes")
                print("\nText: very practical and easy to use")
                print("Reason: app is user-friendly")
                print("\nText: i can not download this zoom app")
                print("Reason: unable to download zoom app")
                continue
            elif not text:
                print("❌ Please enter some text")
                continue
            
            reason = input("🎯 Enter selected reason: ").strip()
            
            if reason.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif not reason:
                print("❌ Please enter a reason")
                continue
            
            # Make prediction
            print("\n🔍 Analyzing...")
            result, confidence, explanation = self.predict_feedback(text, reason)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"Status: {result}")
            print(f"Confidence: {confidence:.1f}%")
            print(f"Explanation: {explanation}")

def main():
    """Main function"""
    print("🚀 Starting EdTech Feedback Validation System...")
    
    # Initialize app with existing model
    app = SimpleFeedbackValidationApp(model_path='feedback_validation_model.pth')
    
    print("\nChoose an option:")
    print("1. Interactive mode (single predictions)")
    print("2. Batch processing (CSV file)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            app.interactive_mode()
            break
        elif choice == '2':
            file_path = input("Enter path to CSV or Excel file: ").strip()
            if file_path:
                result = app.batch_predict(file_path)
                print(result)
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
