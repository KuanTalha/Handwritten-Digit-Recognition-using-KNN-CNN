import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten and fully connected
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class DigitCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Predictor (28x28 Canvas) - CNN Model")

        self.canvas_size = 280
        self.grid_size = 28
        self.cell_size = self.canvas_size // self.grid_size

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load trained CNN model
        try:
            self.model = MNISTModel().to(self.device)
            self.model.load_state_dict(torch.load('mnist_model.pth', map_location=self.device))
            self.model.eval()
            print("Loaded CNN model successfully!")
        except Exception as e:
            print(f"Error: Could not load CNN model. Error: {str(e)}")
            root.destroy()
            return

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack()

        self.image = Image.new("L", (self.grid_size, self.grid_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(pady=5)

        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_rectangle(x, y, x+8, y+8, fill='white', outline='white')

        # Draw to image (scale down to 28x28)
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.draw.rectangle([grid_x, grid_y, grid_x+1, grid_y+1], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.grid_size, self.grid_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="")

    def predict(self):
        # Convert image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Convert PIL image to tensor
        img_tensor = transform(self.image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top 2 predictions
        top2_probs, top2_indices = torch.topk(probabilities, 2)
        top2_predictions = [(idx.item(), prob.item()) for idx, prob in zip(top2_indices, top2_probs)]

        self.prediction_label.config(text=f"Predicted Digit: {top2_predictions[0][0]}")

        # Show pie chart
        labels = [f"{d[0]}" for d in top2_predictions]
        sizes = [d[1] for d in top2_predictions]
        plt.figure(figsize=(4, 4))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        plt.title("Top 2 Predictions")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitCanvasApp(root)
    root.mainloop() 