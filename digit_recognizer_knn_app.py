import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import joblib

class DigitCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Predictor (28x28 Canvas) - KNN Model")

        self.canvas_size = 280
        self.grid_size = 28
        self.cell_size = self.canvas_size // self.grid_size

        # Load trained KNN model
        try:
            self.knn = joblib.load('digit_knn_model.joblib')
            print("Loaded KNN model successfully!")
        except:
            print("Error: Could not load KNN model. Please run digit_knn_trainer.py first.")
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
        # Convert image to array and normalize
        img_array = np.array(self.image).astype("float32") / 255.0
        flat = img_array.flatten().reshape(1, -1)

        # Get predictions using KNN
        predictions = self.knn.predict_proba(flat)[0]
        
        # Get top 2 predictions
        top2_indices = np.argsort(predictions)[-2:][::-1]
        top2_predictions = [(i, predictions[i]) for i in top2_indices]

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