import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os

# Load model path
MODEL_PATH_D = os.path.join("model", "D_model.pth")


# Subcategories
subcategories = ["CRACKED", "UNCRACKED"]

# Define model classes (BasicBlock + ResNet) ... [UNCHANGED]
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = torch.nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channel * block.expansion))
        layers = [block(self.in_channel, out_channel, stride, downsample)]
        self.in_channel = out_channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Last conv layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# === Omit here for brevity - your ResNet class remains the same ===

# Load trained model
def load_model(device="cpu"):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH_D, map_location=device,weights_only=False))
    model.to(device)
    model.eval()
    return model

# Image preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM + heatmap + boxed image + edge detection
def generate_grad_cam_parts(image_pil, model, device="cpu"):
    gradients = []
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    model.layer4.register_forward_hook(hook_fn)

    orig_size = image_pil.size
    image_tensor = data_transforms(image_pil).unsqueeze(0).to(device)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    label = subcategories[pred_class]

    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    act = activations[0].squeeze(0).cpu().detach().numpy()
    grad = gradients[0].squeeze(0).cpu().detach().numpy()

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, orig_size)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

    # Bounding Box image
    image_cv = np.array(image_pil)
    image_boxed = image_cv.copy()
    if pred_class == 0:
        threshold = np.percentile(cam, 90)
        _, binary_mask = cv2.threshold(cam, threshold, 1, cv2.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask * 255)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_boxed, (x, y), (x+w, y+h), (255, 0, 0), 2)
    boxed_img = Image.fromarray(image_boxed)

    # Edge Detection
    edge_img = cv2.Canny(image_cv, 100, 200)
    edge_img = Image.fromarray(edge_img).convert("RGB")

    return label, heatmap, boxed_img, edge_img

# Main Gradio function
def process_image_with_option(image_pil, output_type):
    label, heatmap, boxed_img, edge_img = generate_grad_cam_parts(image_pil, model, device)
    if output_type == "Grad-CAM Heatmap":
        out_img = heatmap
    elif output_type == "Bounding Boxes":
        out_img = boxed_img
    elif output_type == "Edge Detection":
        out_img = edge_img
    else:
        out_img = image_pil
    return out_img, f"Prediction: {label}"

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device)



# Gradio UI
demo = gr.Interface(
    fn=process_image_with_option,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=["Grad-CAM Heatmap", "Bounding Boxes", "Edge Detection"], value="Grad-CAM Heatmap", label="Choose Output Image Type")
    ],
    outputs=[
        gr.Image(type="pil", label="Processed Image"),
        gr.Textbox(label="Prediction Result")
    ],
    title="Crack Detection & Visualization",
    description="Upload an image to check for cracks. Choose the visualization type."
)


#demo.launch()
