from PIL import Image


def load_image(image_path) -> list:
    # Load the image data from the provided image path
    # Implement image loading logic here
    # Return the loaded image data as a list of pixels
    im = Image.open(image_path, "r")
    return list(im.getdata())
