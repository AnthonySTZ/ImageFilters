import image_management as im
import image_filters as filters

if __name__ == "__main__":
    image_data = im.load_image("assets/test/test.jpg")
    filters.blur(image_data, 3)
    filters.blur_cpu(image_data, 3)
    image_data = im.save_image(image_data, "assets/test/new_test.jpg")
