import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(self._get_conv_output((3, 32, 32)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def _get_conv_output(self, shape):
        bs = 1  # Batch size is 1 for testing individual images
        x = torch.rand(bs, *shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x.view(bs, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
net = Net()
net.load_state_dict(torch.load('cifar10_model.pth'))
net.eval()

# Streamlit app
st.title("Image Classification with PyTorch")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image and preprocess it
    image = Image.open(uploaded_file)
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Get model prediction
    with torch.no_grad():
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)

    # Display the image and prediction
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
    st.write(f"Predicted Class: {classes[predicted.item()]}")
