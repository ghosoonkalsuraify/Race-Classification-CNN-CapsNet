import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN


# -----------------------------
# Model Architecture
# -----------------------------
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
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Load Model
# -----------------------------
checkpoint = torch.load("ethnicity_tiny_cnn_capsnet.pth", map_location="cpu")

classes = checkpoint["classes"]

model = TinyCNNCapsNet(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -----------------------------
# CNN-based Face Detector
# -----------------------------
mtcnn = MTCNN(keep_all=True, device="cpu")


# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# -----------------------------
# Face Detection and Cropping
# -----------------------------
def detect_and_crop_face(pil_image):
    boxes, probs = mtcnn.detect(pil_image)

    if boxes is None:
        return pil_image, None

    best_idx = probs.argmax()
    x1, y1, x2, y2 = boxes[best_idx]

    margin = 20
    width, height = pil_image.size

    x1 = max(int(x1) - margin, 0)
    y1 = max(int(y1) - margin, 0)
    x2 = min(int(x2) + margin, width)
    y2 = min(int(y2) + margin, height)

    cropped_face = pil_image.crop((x1, y1, x2, y2))

    return cropped_face, (x1, y1, x2, y2)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(face_image):
    img_tensor = transform(face_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return classes[pred.item()], confidence.item() * 100


# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("AI Face Classification System")
st.write("CNN-based face detection and cropping + CNN-CapsNet classification")

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(img, width=350)

    cropped_face, box = detect_and_crop_face(img)

    if box is None:
        st.warning("No face detected. The model will use the original image.")
    else:
        st.subheader("Detected and Cropped Face")
        st.image(cropped_face, width=250)

    prediction, confidence = predict_image(cropped_face)

    st.success(f"Prediction: {prediction}")
    #st.info(f"Confidence: {confidence:.2f}%")