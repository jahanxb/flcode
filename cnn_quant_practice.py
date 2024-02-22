# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# # Assume model is trained and ready for quantization
# model = SimpleCNN()
# # Specify the quantization configuration
# # In real application, replace this with actual training code and data loading
# model.eval()

# # Fuse Conv, bn and relu
# torch.quantization.fuse_modules(model, [['conv1', 'relu'], ['conv2', 'relu']], inplace=True)

# # Specify quantization configuration
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# # Prepare model for static quantization
# torch.quantization.prepare(model, inplace=True)

# # Calibrate with the representative dataset
# # For demonstration, using a dummy input
# model(torch.randn(1, 1, 28, 28))

# # Convert to quantized model
# torch.quantization.convert(model, inplace=True)

# # Now, the model is quantized and ready for inference
# print(model)

###################################################################################



# import torch
# import torchvision
# import torchvision.transforms as transforms

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# class SimpleCNNForCIFAR(nn.Module):
#     def __init__(self):
#         super(SimpleCNNForCIFAR, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# # Load the pretrained model
# model = SimpleCNNForCIFAR()
# model.eval()  # Set the model to evaluation mode

# # Define quantization configuration
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# # Prepare the model for static quantization
# model_prepared = torch.quantization.prepare(model, inplace=False)

# # Calibrate the prepared model to determine quantization parameters
# # Here we should use a calibration dataset to run through the model, but we'll skip this step for the example

# # Convert the prepared model to a quantized model
# model_quantized = torch.quantization.convert(model_prepared, inplace=False)

# print(model_quantized)


# import torch.optim as optim

# # Assuming model is already defined and loaded
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Training loop
# for epoch in range(10):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the non-quantized model on the 10000 test images: %d %%' % (
#     100 * correct / total))


'''

#############################################


'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import tenseal as ts


# Data transformation and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model definition
class SimpleCNNForCIFAR(nn.Module):
    def __init__(self):
        super(SimpleCNNForCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNNForCIFAR()

print("model: ",model )

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Quantization
#model.eval()
#model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
#model_prepared = torch.quantization.prepare(model, inplace=False)
# model.eval()  # Ensure the model is in eval mode
# with torch.no_grad():
#     for batch, _ in iter(trainloader):
#         model(batch)  # Calibrate with some training data
#         break  # Just a few batches are typically enough for calibration

# model_quantized = torch.quantization.convert(model, inplace=False)

#print('model got quantized here: ',model_quantized)

#model_quantized = torch.quantization.convert(model_prepared, inplace=False)

# Assuming the model is already trained and ready
model.eval()  # Set the model to evaluation mode

# Set the quantization configuration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for quantization
model_prepared = torch.quantization.prepare(model, inplace=False)

# Calibrate the prepared model with some representative data
with torch.no_grad():
    for inputs, _ in trainloader:
        model_prepared(inputs)
        break  # You might want to use more data for calibration

# Convert the prepared and calibrated model to a quantized version
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

# Now, 'model_quantized' is your quantized model

# Evaluation function
def evaluate_model(model_to_eval, loader):
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model_to_eval(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    inference_time = time.time() - start_time
    accuracy = 100 * correct / total
    return accuracy, inference_time

# Evaluating both models
accuracy_non_quantized, time_non_quantized = evaluate_model(model, testloader)
accuracy_quantized, time_quantized = evaluate_model(model_quantized, testloader)

# Printing results
print(f'Non-Quantized Model - Accuracy: {accuracy_non_quantized}%, Inference Time: {time_non_quantized} seconds')
print(f'Quantized Model - Accuracy: {accuracy_quantized}%, Inference Time: {time_quantized} seconds')

print('plain Model: ',model)
print("quantized Model: ",model_quantized)


param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print(' Plain model size: {:.3f}MB'.format(size_all_mb))

param_size = 0
for param in model_quantized.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model_quantized.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print(' model_quantized model size: {:.3f}MB'.format(size_all_mb))