import numpy as np
import torch
import torch.nn as nn
import torch.quantization

# Assuming np_array is your Numpy array with high-precision values
print("Original Numpy Array:")
np_array = np.array([[2.20704021, 0.98369789, 3.00232489],
                     [3.13646733, 1.75611087, 1.39061042]])
print(np_array)

# Convert Numpy array to PyTorch tensor
tensor = torch.tensor(np_array, dtype=torch.float32)

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
    
    def forward(self, x):
        return x

# Initialize dummy model and set it to evaluation mode
model = DummyModel()
model.eval()

# Apply a quantization configuration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for quantization
model_prepared = torch.quantization.prepare(model, inplace=False)

# Quantize and dequantize the tensor
with torch.no_grad():
    # Simulate calibration by passing data through the prepared model
    _ = model_prepared(tensor)
    # Convert the prepared model to a quantized version
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    # Applying the quantized model to the tensor to simulate quantization
    quantized_output = model_quantized(tensor)

# Dequantize the output to get back to floating point representation
dequantized_tensor = quantized_output.dequantize()

print("Simulated 'Quantized' Numpy Array:")
print(dequantized_tensor.numpy())
