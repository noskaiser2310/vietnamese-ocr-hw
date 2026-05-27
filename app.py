import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import os

from src.models.crnn import CRNN
from src.data.vocab import Vocab
from src.utils.metrics import ctc_decode

# Config paths
CKPT_PATH = "checkpoints/nb01_crnn_best.pt"
VOCAB_PATH = "checkpoints/vocab.json"

# Load model globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    vocab = Vocab(VOCAB_PATH)
    model = CRNN(vocab.V).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if 'state' in ckpt:
        model.load_state_dict(ckpt['state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    vocab, model = None, None

def predict(image):
    if model is None or vocab is None:
        return "Error: Model or Vocab not found in checkpoints/ folder."
    
    # Preprocess
    img = image.convert('RGB')
    w, h = img.size
    nw = max(128, min(int(128 * w / h), 2048))
    img = img.resize((nw, 128), Image.Resampling.LANCZOS)
    
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    tensor = tf(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        lp, _ = model(tensor)
        preds = ctc_decode(lp, vocab, blank=0)
        
    return preds[0]

# Setup Gradio Interface
example_images = []
if os.path.exists("test_samples/sample_1.jpg"):
    example_images.append(["test_samples/sample_1.jpg"])

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Vietnamese Handwritten Text Recognition (HTR)")
    gr.Markdown("An end-to-end deep learning model (CRNN + CTC) trained to recognize Vietnamese handwriting with high accuracy. Upload an image to test the model.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Handwritten Image")
            submit_btn = gr.Button("Recognize Text", variant="primary")
        with gr.Column():
            text_output = gr.Textbox(label="Predicted Text", lines=3, show_copy_button=True)
            
    if example_images:
        gr.Examples(examples=example_images, inputs=image_input)
        
    submit_btn.click(fn=predict, inputs=image_input, outputs=text_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
