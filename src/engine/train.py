import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_driver_model
import os

# --- TUNABLE MACRO SETTINGS ---
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

def main():
    # 1. Smart Hardware Detection
    if torch.backends.mps.is_available():
        # Uses your MacBook's M-series silicon chip
        device = torch.device("mps") 
        print("🚀 Using Apple Silicon MPS (Metal Performance Shaders) for Hardware Acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA CUDA GPU for Hardware Acceleration!")
    else:
        # Fallback for older Intel Macs without discrete GPUs
        device = torch.device("cpu") 
        print("🐢 Warning: Only Standard CPU available. Computations will be slow.")
        
    print("\nPreparing Data Loaders...")
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 2. Set Up Engine
    model = get_driver_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n==============================================")
    print("        STARTING TRAINING PIPELINE")
    print("==============================================\n")
    
    best_val_loss = float('inf')
    
    # 3. Main Training Loop
    for epoch in range(EPOCHS):
        model.train() # Set to train mode
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # Clear gradients
            outputs = model(images) # Make prediction
            loss = criterion(outputs, labels) # Calculate error margin
            loss.backward() # Backpropagation (The actual 'learning')
            optimizer.step() # Update weights
            
            running_loss += loss.item()
            
            # Print update every 20 batches
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss margin: {loss.item():.4f}")
                
        # 4. Validation Loop (Test on unseen data to prevent memorization cheating)
        model.eval() # Set to rigid testing mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"\n✅ --- Epoch {epoch+1} Completed ---")
        print(f"Average Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f} | True Accuracy: {val_accuracy:.2f}%\n")
        
        # 5. Save the Brain (Weights)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("./models", exist_ok=True)
            save_path = "./models/driver_vision_v1.pth"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New Best Intelligence Model saved to {save_path}\n")

if __name__ == "__main__":
    main()
