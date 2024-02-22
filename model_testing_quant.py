import torch
import torch.nn as nn
import torch.quantization

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc(x)

model = TestModel()
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model, inplace=False)
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

print(model_quantized)
