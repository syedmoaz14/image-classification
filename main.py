import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Data Loading and Transformation
# Define a transformation for data normalization
transform = transforms.Compose([transforms.ToTensor(),  # Convert to Tensor
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize

# Load the CIFAR-10 training and test data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Define the classes for CIFAR-10
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize the image
    npimg = img.numpy()  # Convert tensor to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert tensor to numpy for plotting
    plt.show()

# Step 2: Model Definition (CNN Architecture)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layer 1: 3 input channels (RGB), 6 output channels (filters), 5x5 kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Convolutional layer 2: 6 input channels, 16 output channels (filters), 5x5 kernel size
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layer 1: Flattened output from conv2, output 120 units
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Fully connected layer 2: 120 units to 84 units
        self.fc2 = nn.Linear(120, 84)
        # Fully connected layer 3: 84 units to 10 output units (for the 10 classes)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Pass the input through the network
        x = F.relu(self.conv1(x))  # Apply ReLU activation after convolution
        x = F.max_pool2d(x, 2, 2)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply ReLU activation after second convolution
        x = F.max_pool2d(x, 2, 2)  # Apply max pooling
        x = x.view(-1, 16 * 5 * 5)  # Flatten the output
        x = F.relu(self.fc1(x))  # Apply ReLU activation after fully connected layer 1
        x = F.relu(self.fc2(x))  # Apply ReLU activation after fully connected layer 2
        x = self.fc3(x)  # Output layer (no activation function here)
        return x

# Step 3: Initialize the Network
net = Net()

# Step 4: Loss Function and Optimizer
# Define the loss function (CrossEntropyLoss is commonly used for classification)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Step 5: Training the Model
for epoch in range(1):  # Number of epochs to train the model
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()   # Zero the parameter gradients
        outputs = net(inputs)   # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()         # Backpropagation
        optimizer.step()        # Optimize the model

        running_loss += loss.item()
        if i % 2000 == 1999:    # Print every 2000 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Step 6: Testing the Model
correct = 0
total = 0

# Test the model on the test data
with torch.no_grad():  # No need to track gradients during testing
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print the accuracy
accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy}%')

# Step 7: Save the Model
torch.save(net.state_dict(), 'cifar10_model.pth')

# Step 8: Load the Model (optional, for future use)
net = Net()  # Create the model again
net.load_state_dict(torch.load('cifar10_model.pth'))  # Load the saved weights
