from PIL import Image
import multiprocessing
from multiprocessing import Process
import timechecking as tcheck


def get_table_pixel(image: Image) -> list:
    width, height = image.size
    image_data = list(image.getdata())
    image_table = []
    for y in range(height):
        image_table.append([])
        for x in range(width):
            image_table[y].append(image_data[y * width + x])
    return image_table


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
    image_table = get_table_pixel(image)
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
    image_table = get_table_pixel(image)

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
    # print(returned_dict)
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

        # print(f"{y} / {height}")

    returned_dict[curr_process] = filtered_data
