from image_management import load_image

if __name__ == "__main__":
    image_data = load_image("assets/test/test.jpg")
    print(image_data[0])
