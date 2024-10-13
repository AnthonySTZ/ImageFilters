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
def box_blur_by_convolution(image: Image, blur_radius: int, multiprocess: bool) -> None:
    blur_kernel = Matrix(
        [[1.0 for _ in range(blur_radius * 2 + 1)] for _ in range(blur_radius * 2 + 1)]
    )
    table_pixels = conv.mult_image_convolve(image, blur_kernel, multiprocess)
    image.putdata(table_pixels)


@tcheck.mesure_function_time
def sharpen_by_convolution(image: Image, strength: int, multiprocess: bool) -> None:
    sharpen_kernel = Matrix(
        [[-1.0, -1.0, -1.0], [-1.0, strength, -1.0], [-1.0, -1.0, -1.0]]
    )

    table_pixels = conv.mult_image_convolve(image, sharpen_kernel, multiprocess)
    image.putdata(table_pixels)


@tcheck.mesure_function_time
def gaussian_blur_by_convolution(
    image: Image, blur_radius: int, multiprocess: bool
) -> None:
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
    table_pixels = conv.mult_image_convolve(image, gaussian_kernel, multiprocess)
    image.putdata(table_pixels)


@tcheck.mesure_function_time
def canny_edge_detector(image: Image, multiprocess: bool) -> None:
    greyscale(image)
    sobel_x_kernel = Matrix([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    sobel_y_kernel = Matrix([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    gradient_x = conv.mult_image_convolve(image, sobel_x_kernel, multiprocess)
    gradient_y = conv.mult_image_convolve(image, sobel_y_kernel, multiprocess)

    gradient_magnitude, gradient_angle = calc_gradient_magnitude_and_angle(
        gradient_x, gradient_y
    )

    edge_threshold = non_maximum_supression(
        gradient_magnitude, gradient_angle, image.size
    )

    threshold_pixels = double_threshold(edge_threshold, image.size)

    pixels = [(int(pixel), int(pixel), int(pixel)) for pixel in threshold_pixels]

    image.putdata(pixels)


def calc_gradient_magnitude_and_angle(
    gradient_x: list, gradient_y: list
) -> tuple[list[float]]:
    gradient_magnitude = [
        math.sqrt(gradient_x[i][0] ** 2 + gradient_y[i][0] ** 2)
        for i in range(len(gradient_x))
    ]
    gradient_angle = [0 for _ in range(len(gradient_x))]

    for i in range(len(gradient_x)):
        angle = math.degrees(
            math.atan2(
                gradient_y[i][0],
                gradient_x[i][0],
            )
        )
        if angle > 157.5:
            angle = 0
        else:
            angle = min([0, 45, 90, 135], key=lambda x: abs(x - angle))

        gradient_angle[i] = angle

    return gradient_magnitude, gradient_angle


def non_maximum_supression(
    gradient_magnitude: list, gradient_angle: list, image_shape: tuple
) -> list[tuple]:
    width, height = image_shape[0], image_shape[1]

    directions = {
        0: [(-1, 0), (1, 0)],
        45: [(1, -1), (-1, 1)],
        90: [(0, -1), (0, 1)],
        135: [(-1, -1), (1, 1)],
    }

    edge_pixels = []

    for y in range(height):
        for x in range(width):
            magnitude = gradient_magnitude[y * width + x]
            angle = gradient_angle[y * width + x]
            on_edge = magnitude
            for direction in directions[angle]:
                dx, dy = direction
                new_x, new_y = x + dx, y + dy
                if 0 < new_x < width and 0 < new_y < height:
                    if magnitude < gradient_magnitude[new_y * width + new_x]:
                        on_edge = 0
            edge_pixels.append(on_edge)

    return edge_pixels


def double_threshold(pixels: list[float], image_shape) -> list[tuple]:
    high_threshold_ratio = 0.7
    low_threshold_ratio = 0.3
    high_threshold = max(pixels) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    new_pixels = [0 for _ in range(len(pixels))]

    width, height = image_shape[0], image_shape[1]

    for y in range(height):
        for x in range(width):
            pixel_value = pixels[y * width + x]
            if pixel_value >= high_threshold:
                new_pixels[y * width + x] = 255
            elif pixel_value < low_threshold:
                pixels[y * width + x] = 255
            else:
                new_pixels[y * width + x] = 0
                for row in range(-1, 2):
                    for col in range(-1, 2):
                        if 0 <= x + col < width and 0 <= y + row < height:
                            if pixel_value < pixels[y * width + row * width + x + col]:
                                new_pixels[y * width + x] = 0
                            else:
                                new_pixels[y * width + x] = 255

    return new_pixels


@tcheck.mesure_function_time
def emboss_by_convolution(image: Image, multiprocess: bool) -> None:
    emboss_kernel = Matrix([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])

    table_pixels = conv.mult_image_convolve(image, emboss_kernel, multiprocess)
    image.putdata(table_pixels)


@tcheck.mesure_function_time
def outline_by_convolution(image: Image, multiprocess: bool) -> None:
    outline_kernel = Matrix([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])

    table_pixels = conv.mult_image_convolve(image, outline_kernel, multiprocess)
    image.putdata(table_pixels)
