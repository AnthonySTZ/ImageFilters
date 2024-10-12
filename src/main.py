import image_management as im
import image_filters as filters
from matrix import Matrix
from image_convolution import image_convolve

if __name__ == "__main__":
    image_data = im.load_image("assets/test/test.jpg")
    # # filters.blur_optimize(image_data, 3)
    # filters.gaussian_blur_optimize(image_data, 5)
    # gaussian_kernel = Matrix([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]])
    # sharpen_kernel = Matrix([[-1.0, -1.0, -1.0], [-1.0, 5.0, -1.0], [-1.0, -1.0, -1.0]])
    # filters.box_blur_by_convolution(image_data, 2)
    filters.gaussian_blur_by_convolution(image_data, 2)
    image_data = im.save_image(image_data, "assets/test/filtered.jpg")
