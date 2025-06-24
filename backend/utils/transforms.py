from PIL import Image
from torchvision import transforms

img_size = (224, 224)

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# Dành cho ảnh PIL
def preprocess_image(image):
    tensor = transform(image)
    return tensor.unsqueeze(0)

