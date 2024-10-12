from PIL import Image
from matrix import Matrix


def get_table_pixel(image: Image) -> list:
    width, height = image.size
    image_data = list(image.getdata())
    image_table = []
    for y in range(height):
        image_table.append([])
        for x in range(width):
            image_table[y].append(image_data[y * width + x])
    return image_table


def image_convolve(image: Image, kernel: Matrix) -> None:
    table_pixel = get_table_pixel(image)
    width, height = len(table_pixel[0]), len(table_pixel)
    padding = int((kernel.size[0] - 1) * 0.5), int((kernel.size[1] - 1) * 0.5)
    convolved_table = []

    print(padding)

    for y in range(height):
        for x in range(width):
            result_r, result_g, result_b = table_pixel[y][x]
            if (
                padding[1] < y < height - padding[1]
                and padding[0] < x < width - padding[0]
            ):
                matrix_r = [[] for _ in range(kernel.size[0])]
                matrix_g = [[] for _ in range(kernel.size[0])]
                matrix_b = [[] for _ in range(kernel.size[0])]
                for row in range(kernel.size[0]):
                    for col in range(kernel.size[1]):
                        matrix_r[row].append(
                            table_pixel[y + row - padding[0]][x + col - padding[1]][0]
                        )
                        matrix_g[row].append(
                            table_pixel[y + row - padding[0]][x + col - padding[1]][1]
                        )
                        matrix_b[row].append(
                            table_pixel[y + row - padding[0]][x + col - padding[1]][2]
                        )
                result_r = Matrix(matrix_r).convolve_by(kernel)
                result_g = Matrix(matrix_g).convolve_by(kernel)
                result_b = Matrix(matrix_b).convolve_by(kernel)

            convolved_table.append((result_r, result_g, result_b))

        print(f"Row {y} / {height - padding[1]}")

    image.putdata(convolved_table)
