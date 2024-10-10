from PIL import Image


def load_image(image_path) -> list:
    # Load the image data from the provided image path
    # Return the loaded image data as a list of pixels

    return Image.open(image_path, "r")


def save_image(image_obj: Image, output_path: str) -> None:
    # Convert the pixels into an array using numpy
    image = Image.new("RGB", image_obj.size)
    image.putdata(image_obj.getdata())
    image.save(output_path)
