# ==========================================================
# GRADIO DEMO FOR AEELR
# Interactive web interface for knee OA classification
# ==========================================================

import os
import numpy as np
import tensorflow as tf
import gradio as gr
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from aeelr_config import CFG
from data_preprocessing import preprocess_pipeline
from explainability import get_gradcam_heatmap
from calibration import TemperatureScaling


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model_and_temperature(model_path, temperature_path=None):
    """
    Load trained model and temperature scaler
    
    Args:
        model_path: Path to saved model (.h5)
        temperature_path: Path to temperature file (.npy)
    
    Returns:
        (model, temp_scaler)
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    temp_scaler = None
    if temperature_path and os.path.exists(temperature_path):
        print(f"Loading temperature from {temperature_path}...")
        temp_scaler = TemperatureScaling()
        temp_scaler.load(temperature_path)
    
    return model, temp_scaler


# Global model and temperature scaler
MODEL = None
TEMP_SCALER = None


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_koa_grade(image):
    """
    Predict KOA grade from uploaded image
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        (prediction_text, gradcam_overlay, confidence_plot)
    """
    global MODEL, TEMP_SCALER
    
    if MODEL is None:
        return "❌ Model not loaded!", None, None
    
    try:
        # Save temporary image
        temp_path = "temp_upload.png"
        if isinstance(image, np.ndarray):
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            image.save(temp_path)
        
        # Preprocess
        img = preprocess_pipeline(temp_path, target_size=CFG.IMG_SIZE, return_rgb=True)
        img_array = np.expand_dims(img, axis=0)
        
        # Predict
        preds = MODEL.predict(img_array, verbose=0)
        
        # Handle hierarchical outputs
        if isinstance(preds, list):
            binary_pred, ternary_pred, kl_pred = preds
        else:
            kl_pred = preds
            binary_pred = None
            ternary_pred = None
        
        # Apply temperature scaling
        if TEMP_SCALER is not None:
            kl_probs = TEMP_SCALER.predict(kl_pred)
        else:
            kl_probs = tf.nn.softmax(kl_pred).numpy()
        
        # Get prediction
        pred_class = np.argmax(kl_probs[0])
        confidence = kl_probs[0][pred_class]
        
        # Build prediction text
        prediction_text = f"## Predicted KL Grade: **{CFG.CLASS_NAMES[pred_class]}**\n\n"
        prediction_text += f"### Confidence: **{confidence:.2%}**\n\n"
        
        # Add hierarchical predictions if available
        if binary_pred is not None:
            binary_probs = tf.nn.softmax(binary_pred).numpy()[0]
            prediction_text += f"#### Binary Classification:\n"
            prediction_text += f"- Healthy: {binary_probs[0]:.2%}\n"
            prediction_text += f"- OA: {binary_probs[1]:.2%}\n\n"
        
        if ternary_pred is not None:
            ternary_probs = tf.nn.softmax(ternary_pred).numpy()[0]
            prediction_text += f"#### Severity Classification:\n"
            prediction_text += f"- Mild: {ternary_probs[0]:.2%}\n"
            prediction_text += f"- Moderate: {ternary_probs[1]:.2%}\n"
            prediction_text += f"- Severe: {ternary_probs[2]:.2%}\n\n"
        
        # Add all class probabilities
        prediction_text += f"#### All KL Grades:\n"
        for i, class_name in enumerate(CFG.CLASS_NAMES):
            prediction_text += f"- {class_name}: {kl_probs[0][i]:.2%}\n"
        
        # Generate Grad-CAM
        try:
            heatmap = get_gradcam_heatmap(MODEL, img_array, CFG.GRADCAM_LAYER, pred_class)
            
            # Load original image
            original = cv2.imread(temp_path)
            original = cv2.resize(original, CFG.IMG_SIZE)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            heatmap_resized = cv2.resize(heatmap, CFG.IMG_SIZE)
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
            
            gradcam_overlay = Image.fromarray(overlay)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            gradcam_overlay = None
        
        # Create confidence plot
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['green' if i == pred_class else 'steelblue' for i in range(CFG.NUM_CLASSES)]
        ax.barh(CFG.CLASS_NAMES, kl_probs[0], color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_title('KL Grade Confidence Scores', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        confidence_plot = Image.open(buf)
        plt.close()
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return prediction_text, gradcam_overlay, confidence_plot
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


# ==========================================================
# GRADIO INTERFACE
# ==========================================================

def create_demo(model_path, temperature_path=None):
    """
    Create Gradio demo interface
    
    Args:
        model_path: Path to trained model
        temperature_path: Path to temperature file
    
    Returns:
        Gradio Blocks interface
    """
    global MODEL, TEMP_SCALER
    
    # Load model
    MODEL, TEMP_SCALER = load_model_and_temperature(model_path, temperature_path)
    
    # Create interface
    with gr.Blocks(title="AEELR: Knee OA Classification", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🦴 AEELR: Attention-Enhanced EfficientNet for Knee Osteoarthritis Classification
        
        Upload a knee X-ray image to get:
        - **KL Grade Prediction** (0-4)
        - **Confidence Scores** for all grades
        - **Grad-CAM Visualization** showing which regions influenced the prediction
        - **Hierarchical Classification** (Healthy vs OA, Severity levels)
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Knee X-ray",
                    type="pil",
                    height=400
                )
                
                predict_btn = gr.Button("🔍 Predict KL Grade", variant="primary", size="lg")
                
                gr.Markdown("""
                ### About AEELR
                
                This model uses:
                - **EfficientNetB5** backbone (ImageNet pretrained)
                - **CBAM Attention** for feature refinement
                - **Temperature Scaling** for calibrated confidence
                - **Grad-CAM** for interpretability
                
                **KL Grades:**
                - KL-0: Healthy
                - KL-1: Doubtful
                - KL-2: Minimal
                - KL-3: Moderate
                - KL-4: Severe
                """)
            
            with gr.Column(scale=2):
                prediction_output = gr.Markdown(label="Prediction")
                
                with gr.Row():
                    gradcam_output = gr.Image(label="Grad-CAM Overlay", height=300)
                    confidence_output = gr.Image(label="Confidence Scores", height=300)
        
        # Connect button
        predict_btn.click(
            fn=predict_koa_grade,
            inputs=image_input,
            outputs=[prediction_output, gradcam_output, confidence_output]
        )
        
        gr.Markdown("""
        ---
        ### ⚠️ Disclaimer
        
        This is a research prototype for educational purposes only. 
        **NOT** intended for clinical diagnosis. Always consult a qualified healthcare professional.
        """)
    
    return demo


# ==========================================================
# MAIN
# ==========================================================

def main():
    """Launch Gradio demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEELR Gradio Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.h5)")
    parser.add_argument("--temperature", type=str, default=None, help="Path to temperature file (.npy)")
    parser.add_argument("--port", type=int, default=CFG.DEMO_PORT, help="Port to run demo on")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = create_demo(args.model, args.temperature)
    
    print("\n" + "="*70)
    print("🚀 LAUNCHING AEELR DEMO")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Port: {args.port}")
    print("="*70 + "\n")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
