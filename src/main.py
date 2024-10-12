import image_management as im
import image_filters as filters
from matrix import Matrix

if __name__ == "__main__":
    # image_data = im.load_image("assets/test/test.jpg")
    # # filters.blur_optimize(image_data, 3)
    # filters.gaussian_blur_optimize(image_data, 5)
    # image_data = im.save_image(image_data, "assets/test/filtered.jpg")
    matrix = Matrix([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]])
    matrix_2 = Matrix([[1.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 3.0]])
    matrix_3 = Matrix([[]])
    print(matrix)
    print(matrix_2)
