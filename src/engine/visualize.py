import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the python path so we can import our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_driver_model

def visualize_feature_maps():
    print("🧠 Initializing Deep Learning Visualization Engine...")
    
    # 1. Load the model
    model = get_driver_model(num_classes=10)
    model.eval() # Set to evaluation mode
    
    # 2. Simulate an incoming driver image (1 channel grayscale, 224x224)
    # In a real scenario, this would be `image = Image.open('driver.jpg')`
    print("📸 Simulating a 224x224 Driver Image Input Tensor...")
    dummy_image = torch.randn(1, 1, 224, 224)
    
    # 3. Extract Intermediate Layer Activations (The Black Box internals)
    print("🔍 Extracting Convolutional Feature Maps...")
    
    with torch.no_grad():
        # Block 1 Activation
        x1 = model.pool1(F.relu(model.conv1(dummy_image)))
        
        # Block 2 Activation
        x2 = model.pool2(F.relu(model.conv2(x1)))
        
        # Block 3 Activation
        x3 = model.pool3(F.relu(model.conv3(x2)))
        
    print("✅ Feature extraction complete. Generating plots...")
    
    # 4. Plotting
    try:
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        fig.suptitle('Neural Network "Black Box" Visibility: What the AI is seeing', fontsize=16)
        
        # We plot the first 4 channels of each Block to see different edge/texture derivations
        blocks = [
            ("Block 1 Output (Raw Edge Detection)", x1),
            ("Block 2 Output (Spatial Grouping)", x2),
            ("Block 3 Output (Deep Abstraction)", x3)
        ]
        
        for row, (title, tensor) in enumerate(blocks):
            # Extract the numpy array of the shape (Batch, Channels, Height, Width)
            feature_maps = tensor[0].cpu().numpy()
            
            for col in range(4):
                ax = axes[row, col]
                # Plot the specific channel from the feature map
                if col < feature_maps.shape[0]:
                    ax.imshow(feature_maps[col], cmap='viridis')
                ax.axis('off')
                
                # Add title only to the first column for clarity
                if col == 0:
                    ax.set_title(title, loc='left', pad=10, fontsize=12)
                    
        plt.tight_layout()
        
        # Create an output directory for visuals
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        visuals_dir = os.path.join(base_dir, "docs", "visuals")
        os.makedirs(visuals_dir, exist_ok=True)
        
        save_path = os.path.join(visuals_dir, "cnn_feature_maps.png")
        
        # Resolve absolute path for clarity in logging
        abs_path = os.path.abspath(save_path)
        plt.savefig(abs_path)
        
        print(f"\n🎉 SUCCESS! The internal visual thought-process of the AI has been saved.")
        print(f"📁 You can find the presentation image at: {abs_path}")
        print("Show this image to your mentor to explain exactly how the convolutional layers process visual textures!")
        
    except ImportError:
        print("\n❌ Matplotlib is not installed! Run `pip install matplotlib` to generate the image.")

if __name__ == "__main__":
    visualize_feature_maps()
