import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import io
import sys
import os

# Add src/engine to python path so we can import our custom model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'engine')))
try:
    from model import get_driver_model
except ImportError:
    st.error("Error: Could not import CustomDriverCNN. Check your project structure.")

# Set page config
st.set_page_config(
    page_title="Smart Road Safety AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State Farm Dataset Classes
CLASS_NAMES = {
    0: "Safe Driving 🟢",
    1: "Texting - Right 🔴",
    2: "Talking on Phone - Right 🔴",
    3: "Texting - Left 🔴",
    4: "Talking on Phone - Left 🔴",
    5: "Operating the Radio 🟡",
    6: "Drinking 🟡",
    7: "Reaching Behind 🔴",
    8: "Hair and Makeup 🔴",
    9: "Talking to Passenger 🟡",
}

@st.cache_resource
def load_model():
    """Loads the Custom CNN model and its weights"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = get_driver_model(num_classes=10)
    
    weights_path = "./models/driver_vision_v1.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        st.sidebar.success("✅ Model Weights Loaded Successfully")
    else:
        st.sidebar.warning("⚠️ Warning: Pre-trained weights not found at ./models/driver_vision_v1.pth. Running entirely on randomized initialized weights (Demo Mode).")
    
    model.to(device)
    model.eval()
    return model, device

def process_image(image):
    """Applies the same transformations used during training to prevent dataset shift"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        # Convert RGB to Grayscale across 1 channel exactly like the CNN expects
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch

def generate_feature_maps(model, image_tensor, device):
    """Extracts internal layer activations for visualization"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        x1 = model.pool1(F.relu(model.conv1(image_tensor)))
        x2 = model.pool2(F.relu(model.conv2(x1)))
        x3 = model.pool3(F.relu(model.conv3(x2)))
        
    fig, axes = plt.subplots(3, 4, figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117') # Streamlit dark mode background
    
    blocks = [
        ("Block 1 Output (Raw Edge Detection)", x1),
        ("Block 2 Output (Spatial Grouping)", x2),
        ("Block 3 Output (Deep Abstraction)", x3)
    ]
    
    for row, (title, tensor) in enumerate(blocks):
        feature_maps = tensor[0].cpu().numpy()
        for col in range(4):
            ax = axes[row, col]
            if col < feature_maps.shape[0]:
                ax.imshow(feature_maps[col], cmap='viridis')
            ax.axis('off')
            if col == 0:
                ax.set_title(title, loc='left', pad=10, fontsize=10, color='white')
                
    plt.tight_layout()
    return fig

# --- UI LAYOUT ---

st.title("🚗 Smart Road Safety & Driving Intelligence")
st.markdown("### Edge AI Live Prototype")
st.write("Upload an image from a cabin-facing dashcam (or a photo of yourself acting as a driver) to test the Driver Monitoring System (DMS) inference pipeline in real-time.")

st.sidebar.title("System Status")
st.sidebar.info("Model: CustomDriverCNN\n\nBackend: PyTorch\n\nOptimization: Raw Inference")
model, device = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Video Frame Input")
    uploaded_file = st.file_uploader("Upload driver cabin frame...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Processed Cabin Frame', use_container_width=True)

with col2:
    st.subheader("2. AI Telemetry Output")
    if uploaded_file is not None:
        with st.spinner('Running AI Inference Engine...'):
            # Preprocess
            input_tensor = process_image(image)
            image_tensor = input_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = F.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            conf_pct = confidence.item() * 100
            
            # Display Prediction
            st.markdown("### Decision Matrix:")
            
            if "Safe" in predicted_class:
                st.success(f"**{predicted_class}** (Confidence: {conf_pct:.2f}%)")
            elif "🟡" in predicted_class:
                st.warning(f"**{predicted_class}** (Confidence: {conf_pct:.2f}%)")
            else:
                st.error(f"**{predicted_class}** (Confidence: {conf_pct:.2f}%)")
            
            st.progress(int(conf_pct))
            
            st.markdown("---")
            st.markdown("### 🔍 Model Explainability (Neural X-Ray)")
            if st.button("Generate Diagnostic Feature Maps"):
                with st.spinner("Extracting layer activations..."):
                    fig = generate_feature_maps(model, input_tensor, device)
                    st.pyplot(fig)
                    st.caption("These visual heatmaps show exactly geometric shapes the internal Pytorch Convolution blocks are isolating to make their decision.")
    else:
        st.info("Awaiting video frame upload to begin AI Pipeline.")
