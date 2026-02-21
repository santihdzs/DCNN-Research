# Example training script

import torch
from src.models import get_model, count_parameters
from src.pipeline import get_optimizer, get_criterion, train_epoch, evaluate, save_checkpoint
from src.utils import get_mnist_loaders

def main():
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
    
    # Create model
    print("Creating model...")
    model = get_model(device=DEVICE)
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Setup training
    criterion = get_criterion()
    optimizer = get_optimizer(model, optimizer="adam", lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Evaluate on test set
        test_results = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_results['loss']:.4f} | Test Acc:  {test_results['accuracy']:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_results["loss"],
                "test_acc": test_results["accuracy"]
            })
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
