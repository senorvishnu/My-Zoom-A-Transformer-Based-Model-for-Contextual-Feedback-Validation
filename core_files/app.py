import gradio as gr
import torch
import pandas as pd
import numpy as np
from model import FeedbackValidationModel
from data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class FeedbackValidationApp:
    def __init__(self, model_path: str = None):
        self.model = FeedbackValidationModel()
        self.preprocessor = DataPreprocessor()
        
        # Load model if path is provided
        if model_path:
            try:
                self.model.load_model(model_path)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
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
    
    def batch_predict(self, file) -> str:
        """
        Process batch predictions from uploaded CSV file
        """
        if file is None:
            return "Please upload a CSV file with 'text' and 'reason' columns."
        
        try:
            # Read CSV file
            df = pd.read_csv(file.name)
            
            if 'text' not in df.columns or 'reason' not in df.columns:
                return "CSV file must contain 'text' and 'reason' columns."
            
            results = []
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
            **Batch Prediction Results:**
            
            - Total samples: {total_count}
            - Aligned feedback: {aligned_count}
            - Not aligned feedback: {total_count - aligned_count}
            - Average confidence: {avg_confidence:.1f}%
            
            Results saved to: {output_path}
            """
            
            return summary
            
        except Exception as e:
            return f"Error processing batch file: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .main-header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .prediction-box {
            border: 2px solid #3498db;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .confidence-bar {
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            height: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css, title="EdTech Feedback Validation") as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🎓 EdTech Feedback Validation System</h1>
                <p>Validate whether user feedback aligns with selected dropdown reasons using AI</p>
            </div>
            """)
            
            with gr.Tab("Single Prediction"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📝 Input Feedback</h3>")
                        
                        text_input = gr.Textbox(
                            label="User Feedback Text",
                            placeholder="Enter the user's feedback here...",
                            lines=4,
                            max_lines=6
                        )
                        
                        reason_input = gr.Textbox(
                            label="Selected Reason",
                            placeholder="Enter the dropdown reason selected by the user...",
                            lines=2,
                            max_lines=3
                        )
                        
                        predict_btn = gr.Button("🔍 Validate Feedback", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h3>🎯 Prediction Results</h3>")
                        
                        result_output = gr.Textbox(
                            label="Alignment Status",
                            interactive=False,
                            elem_classes=["prediction-box"]
                        )
                        
                        confidence_output = gr.Slider(
                            label="Confidence Score",
                            minimum=0,
                            maximum=100,
                            interactive=False,
                            elem_classes=["confidence-bar"]
                        )
                        
                        explanation_output = gr.Textbox(
                            label="Explanation",
                            interactive=False,
                            lines=3
                        )
                
                # Example inputs
                gr.HTML("<h4>💡 Example Inputs</h4>")
                with gr.Row():
                    gr.Examples(
                        examples=[
                            ["this is an amazing app for online classes!", "good app for conducting online classes"],
                            ["very practical and easy to use", "app is user-friendly"],
                            ["i can not download this zoom app", "unable to download zoom app"],
                            ["the app crashes when i try to join a meeting", "good app for conducting online classes"],
                            ["this app is terrible for video calls", "good for video conferencing"]
                        ],
                        inputs=[text_input, reason_input],
                        label="Try these examples"
                    )
            
            with gr.Tab("Batch Processing"):
                gr.HTML("<h3>📊 Batch Prediction</h3>")
                gr.HTML("<p>Upload a CSV file with 'text' and 'reason' columns to process multiple feedback entries at once.</p>")
                
                file_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    file_count="single"
                )
                
                batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                
                batch_output = gr.Markdown(label="Batch Results")
            
            with gr.Tab("About"):
                gr.HTML("""
                <div style="padding: 20px;">
                    <h2>About This System</h2>
                    
                    <h3>🎯 Purpose</h3>
                    <p>This AI-powered system validates whether user feedback in EdTech applications aligns with the selected dropdown reasons. It helps ensure data quality and relevance in feedback collection systems.</p>
                    
                    <h3>🔧 How It Works</h3>
                    <ul>
                        <li><strong>Text Processing:</strong> Cleans and preprocesses both feedback text and reason</li>
                        <li><strong>AI Analysis:</strong> Uses BERT-based transformer model to understand context and alignment</li>
                        <li><strong>Classification:</strong> Predicts whether feedback aligns (1) or doesn't align (0) with the reason</li>
                        <li><strong>Confidence Scoring:</strong> Provides confidence levels for each prediction</li>
                    </ul>
                    
                    <h3>📈 Use Cases</h3>
                    <ul>
                        <li>Enhanced feedback systems in EdTech platforms</li>
                        <li>Automated moderation of user feedback</li>
                        <li>Quality control in online surveys</li>
                        <li>Data analytics for course improvement</li>
                    </ul>
                    
                    <h3>🎓 Technical Details</h3>
                    <ul>
                        <li><strong>Model:</strong> BERT-based transformer fine-tuned for text pair classification</li>
                        <li><strong>Accuracy:</strong> Expected to achieve >85% accuracy on validation data</li>
                        <li><strong>Framework:</strong> PyTorch with Transformers library</li>
                        <li><strong>Interface:</strong> Gradio web application</li>
                    </ul>
                </div>
                """)
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_feedback,
                inputs=[text_input, reason_input],
                outputs=[result_output, confidence_output, explanation_output]
            )
            
            batch_btn.click(
                fn=self.batch_predict,
                inputs=[file_input],
                outputs=[batch_output]
            )
        
        return interface

def main():
    """Main function to run the application"""
    print("🚀 Starting EdTech Feedback Validation System...")
    
    # Initialize app with existing model
    app = FeedbackValidationApp(model_path='feedback_validation_model.pth')
    
    # Create and launch interface
    interface = app.create_interface()
    
    print("🌐 Launching web interface...")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()


