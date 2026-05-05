import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class TinyCNNCapsNet(nn.Module):
    def __init__(self, num_classes=3):
        super(TinyCNNCapsNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),   # 🔥 هذا مهم
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# تحميل المودل
checkpoint = torch.load("ethnicity_tiny_cnn_capsnet.pth", map_location="cpu")

model = TinyCNNCapsNet(num_classes=len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

classes = checkpoint["classes"]

# Transform
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# تحميل صورة
img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(img)
    _, pred = torch.max(outputs, 1)

print("Prediction:", classes[pred.item()])