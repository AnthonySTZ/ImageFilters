import image_management as im

if __name__ == "__main__":
    image_data = im.load_image("assets/test/test.jpg")
    image_data = im.save_image(image_data, "assets/test/new_test.jpg")
