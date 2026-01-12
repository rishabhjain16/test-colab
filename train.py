import argparse
import time
import torch

def train(epochs, lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    print(f"Config: epochs={epochs}, lr={lr}")
    
    for epoch in range(epochs):
        time.sleep(1)
        loss = 1.0 / (epoch + 1)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    # Save results
    with open("results.txt", "w") as f:
        f.write(f"Training completed on {device}\n")
        f.write(f"Final loss: {loss:.4f}\n")
    
    print("Done! Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args.epochs, args.lr)

    #Updated
