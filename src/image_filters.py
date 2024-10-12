from PIL import Image
import multiprocessing
from multiprocessing import Process
import timechecking as tcheck
import image_convolution as conv
from matrix import Matrix
import math


def greyscale(image: Image) -> None:
    image_data = list(image.getdata())
    filtered_data = []
    for pixel in image_data:
        r, g, b = pixel
        grey_value = int((r + g + b) * 0.33)
        pixel = (
            grey_value,
            grey_value,
            grey_value,
        )  # Replace pixel with greyscale value
        filtered_data.append(pixel)
    image.putdata(filtered_data)


@tcheck.mesure_function_time
def blur(image: Image, blur_radius: int) -> None:
    image_table = conv.get_table_pixel(image)
    filtered_data = []
    width, height = len(image_table[0]), len(image_table)

    for y in range(height):
        for x in range(width):
            pixel = [0, 0, 0]
            num_pixels = 0
            for radius_y in range(-blur_radius, blur_radius + 1):
                for radius_x in range(-blur_radius, blur_radius + 1):
                    if 0 <= y + radius_y < height and 0 <= x + radius_x < width:
                        pixel[0] += image_table[y + radius_y][x + radius_x][0]
                        pixel[1] += image_table[y + radius_y][x + radius_x][1]
                        pixel[2] += image_table[y + radius_y][x + radius_x][2]
                        num_pixels += 1

            pixel[0] = int(pixel[0] / num_pixels)
            pixel[1] = int(pixel[1] / num_pixels)
            pixel[2] = int(pixel[2] / num_pixels)
            filtered_data.append(tuple(pixel))

        # print(f"{y} / {height}")

    image.putdata(filtered_data)


@tcheck.mesure_function_time
def blur_optimize(image: Image, blur_radius: int) -> None:
    image_table = conv.get_table_pixel(image)

    manager = multiprocessing.Manager()
    returned_dict = manager.dict()

    procs = []
    nb_of_procs = multiprocessing.cpu_count()

    for proc_nb in range(nb_of_procs):
        proc = Process(
            target=mult_proc_blur,
            args=(proc_nb, nb_of_procs, image_table, blur_radius, returned_dict),
        )  # instantiating without any argument
        procs.append(proc)
        proc.start()

    for p in procs:
        p.join()

    filtered_image = []
    for proc_nb in range(nb_of_procs):
        filtered_image.extend(returned_dict[proc_nb])

    image.putdata(filtered_image)


def mult_proc_blur(
    curr_process: int,
    nb_of_processes: int,
    base_image_table: list,
    blur_radius: int,
    returned_dict: dict,
) -> None:

    filtered_data = []
    width, height = len(base_image_table[0]), len(base_image_table)
    height_start = int((height / nb_of_processes) * curr_process)
    height_end = int((height / nb_of_processes) * (curr_process + 1))

    for y in range(height_start, height_end):
        for x in range(width):
            pixel = [0, 0, 0]
            num_pixels = 0
            for radius_y in range(-blur_radius, blur_radius + 1):
                for radius_x in range(-blur_radius, blur_radius + 1):
                    if 0 <= y + radius_y < height and 0 <= x + radius_x < width:
                        pixel[0] += base_image_table[y + radius_y][x + radius_x][0]
                        pixel[1] += base_image_table[y + radius_y][x + radius_x][1]
                        pixel[2] += base_image_table[y + radius_y][x + radius_x][2]
                        num_pixels += 1

            pixel[0] = int(pixel[0] / num_pixels)
            pixel[1] = int(pixel[1] / num_pixels)
            pixel[2] = int(pixel[2] / num_pixels)
            filtered_data.append(tuple(pixel))

    returned_dict[curr_process] = filtered_data


@tcheck.mesure_function_time
def gaussian_blur_optimize(image: Image, blur_radius: int) -> None:
    image_table = conv.get_table_pixel(image)

    manager = multiprocessing.Manager()
    returned_dict = manager.dict()

    procs = []
    nb_of_procs = multiprocessing.cpu_count()

    for proc_nb in range(nb_of_procs):
        proc = Process(
            target=mult_proc_gaussian_blur,
            args=(proc_nb, nb_of_procs, image_table, blur_radius, returned_dict),
        )  # instantiating without any argument
        procs.append(proc)
        proc.start()

    for p in procs:
        p.join()

    filtered_image = []
    for proc_nb in range(nb_of_procs):
        filtered_image.extend(returned_dict[proc_nb])

    image.putdata(filtered_image)


def mult_proc_gaussian_blur(
    curr_process: int,
    nb_of_processes: int,
    base_image_table: list,
    blur_radius: int,
    returned_dict: dict,
) -> None:

    filtered_data = []
    width, height = len(base_image_table[0]), len(base_image_table)
    height_start = int((height / nb_of_processes) * curr_process)
    height_end = int((height / nb_of_processes) * (curr_process + 1))

    pi = 3.1415
    e = 2.718
    standard_deviation = blur_radius / 3

    for y in range(height_start, height_end):
        for x in range(width):
            pixel = [0, 0, 0]
            for radius_y in range(-blur_radius, blur_radius + 1):
                for radius_x in range(-blur_radius, blur_radius + 1):
                    if 0 <= y + radius_y < height and 0 <= x + radius_x < width:
                        exponent = -(
                            (abs(radius_x) ** 2 + abs(radius_y) ** 2)
                            / (2 * standard_deviation * standard_deviation)
                        )
                        gaussian = (
                            1 / (2 * pi * standard_deviation * standard_deviation)
                        ) * pow(e, exponent)

                        pixel[0] += (
                            base_image_table[y + radius_y][x + radius_x][0] * gaussian
                        )
                        pixel[1] += (
                            base_image_table[y + radius_y][x + radius_x][1] * gaussian
                        )
                        pixel[2] += (
                            base_image_table[y + radius_y][x + radius_x][2] * gaussian
                        )

            pixel[0] = int(pixel[0])
            pixel[1] = int(pixel[1])
            pixel[2] = int(pixel[2])
            filtered_data.append(tuple(pixel))

    returned_dict[curr_process] = filtered_data


@tcheck.mesure_function_time
def box_blur_by_convolution(image: Image, blur_radius: int) -> None:
    blur_kernel = Matrix(
        [[1.0 for _ in range(blur_radius * 2 + 1)] for _ in range(blur_radius * 2 + 1)]
    )
    table_pixels = conv.image_convolve(image, blur_kernel)
    image.putdata(table_pixels)


@tcheck.mesure_function_time
def sharpen_by_convolution(image: Image, strength: int) -> None:
    sharpen_kernel = Matrix(
        [[-1.0, -1.0, -1.0], [-1.0, strength, -1.0], [-1.0, -1.0, -1.0]]
    )

    table_pixels = conv.image_convolve(image, sharpen_kernel)
    image.putdata(table_pixels)


@tcheck.mesure_function_time
def gaussian_blur_by_convolution(image: Image, blur_radius: int) -> None:
    gaussian_matrix = []
    for y in range(blur_radius * 2 + 1):
        gaussian_matrix.append([])
        for x in range(blur_radius * 2 + 1):
            distance = (x - blur_radius) ** 2 + (y - blur_radius) ** 2
            weight = (
                1.0
                / (2 * math.pi * blur_radius**2)
                * math.exp(-distance / (2 * blur_radius**2))
            )
            gaussian_matrix[y].append(weight)
    gaussian_kernel = Matrix(gaussian_matrix)
    table_pixels = conv.image_convolve(image, gaussian_kernel)
    image.putdata(table_pixels)


@tcheck.mesure_function_time
def canny_edge_detector(image: Image) -> None:
    greyscale(image)
    sobel_x_kernel = Matrix([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    sobel_y_kernel = Matrix([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    gradient_x = conv.image_convolve(image, sobel_x_kernel)
    gradient_y = conv.image_convolve(image, sobel_y_kernel)

    gradient_magnitude, gradient_angle = calc_gradient_magnitude_and_angle(
        gradient_x, gradient_y
    )

    print(gradient_magnitude[0:5])

    pixels = [(int(pixel), int(pixel), int(pixel)) for pixel in gradient_magnitude]

    image.putdata(pixels)


def calc_gradient_magnitude_and_angle(
    gradient_x: int, gradient_y: int
) -> tuple[list[float]]:
    gradient_magnitude = [
        math.sqrt(gradient_x[i][0] ** 2 + gradient_y[i][0] ** 2)
        for i in range(len(gradient_x))
    ]
    gradient_angle = [
        math.degrees(
            math.atan2(
                gradient_y[i][0],
                gradient_x[i][0],
            )
        )
        for i in range(len(gradient_x))
    ]

    return gradient_magnitude, gradient_angle
