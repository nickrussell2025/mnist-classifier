import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import gradio as gr


class MNISTClassifier(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate=0.25):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        input_size = 784
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTClassifier([256, 128, 64], dropout_rate=0.20).to(device)
model.load_state_dict(torch.load('best_model_final.pth', map_location=device))
model.eval()


def predict_digit(sketch):
    if sketch is None:
        return {}
    
    if isinstance(sketch, dict):
        sketch = sketch.get('composite')
        if sketch is None:
            return {}
    
    if sketch.size == 0:
        return {}
    
    if sketch.ndim == 3:
        sketch = sketch.mean(axis=2)
    
    img_array = sketch.astype(np.float32) / 255.0
    img_array = 1.0 - img_array  # Gradio → MNIST format
    
    # Bounding box
    rows_active = np.any(img_array > 0.1, axis=1)
    cols_active = np.any(img_array > 0.1, axis=0)
    
    if not rows_active.any() or not cols_active.any():
        return {}
    
    rmin, rmax = np.where(rows_active)[0][[0, -1]]
    cmin, cmax = np.where(cols_active)[0][[0, -1]]
    rmin, rmax = max(0, rmin-2), min(img_array.shape[0], rmax+2)
    cmin, cmax = max(0, cmin-2), min(img_array.shape[1], cmax+2)
    
    cropped = img_array[rmin:rmax, cmin:cmax]

    # Aspect-preserving resize
    rows_c, cols_c = cropped.shape
    if rows_c > cols_c:
        new_rows, new_cols = 20, int(cols_c * 20 / rows_c)
    else:
        new_cols, new_rows = 20, int(rows_c * 20 / cols_c)
    
    img = Image.fromarray((cropped * 255).astype('uint8'))
    img = img.resize((new_cols, new_rows))
    resized = np.array(img) / 255.0

    final = np.zeros((28, 28))
    row_offset = (28 - new_rows) // 2
    col_offset = (28 - new_cols) // 2
    final[row_offset:row_offset+new_rows, col_offset:col_offset+new_cols] = resized

    final = (final - 0.1307) / 0.3081
    img_tensor = torch.FloatTensor(final).view(1, 1, 28, 28).to(device)
    
    with torch.inference_mode():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    
    return {str(i): float(probs[i]) for i in range(10)}


demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy", image_mode="L"),
    outputs=gr.Label(num_top_classes=3),
    title="MNIST Digit Classifier",
    description="Draw 0-9",
    live=False
)


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
