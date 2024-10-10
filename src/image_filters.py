from PIL import Image


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

        print(f"{y} / {height}")

    image.putdata(filtered_data)
