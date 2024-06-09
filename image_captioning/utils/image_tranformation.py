import torchvision.transforms as transforms
from PIL import Image


def resize_and_normalize_image(img, width, height, mean, std):
    """
    Takes and image object as input and resizes and normalizes it following the specified parameters.
    Additionally, adds an extra dimension at position 0 corresponding to the batch size.
    """
    resized_img = img.resize([width, height], Image.LANCZOS)

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    resized_and_normalized_img = transformation(resized_img).unsqueeze(0)
    return resized_and_normalized_img
