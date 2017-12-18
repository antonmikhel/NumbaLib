import cv2
import numba
import numpy


@numba.jit(nopython=True, cache=True)
def __numba_interp_bilinear(input_image, output_image):
    width, height, channels = output_image.shape
    in_width, in_height = input_image.shape[:-1]

    w_ratio = width / in_width
    h_ration = height / in_height

    for x in range(width):
        orig_x = float(x) / w_ratio
        i_x = int(orig_x)
        floor_x = orig_x - i_x
        ceil_x = 1 - orig_x
        for y in range(height):
            r = 0.0
            g = 0.0
            b = 0.0

            orig_y = float(y) / w_ratio
            i_y = int(orig_y)
            floor_y = orig_y - i_y
            ceil_y = 1 - orig_y

            mult1 = ceil_x * ceil_y
            mult2 = ceil_x * floor_y
            mult3 = floor_x * ceil_y
            mult3 = floor_x * floor_y

            # r +=


def interp_bilinear(input_image, output_image):
    """
    :Description: This function resizes input_image to output_image dimensions using bilinear interpolation
    :param input_image: numpy array, same number of channels as output image
    :type input_image: numpy.ndarray
    :param output_image: numpy array, same number of elements as input image
    :type output_image: numpy.ndarray
    :return: None
    """

    __numba_interp_bilinear(input_image, output_image)


def __test_interp():

    input_image_path = r"D:\Dev\NumbaLib\Assets\Peter.jpg"
    resize_factor = 5

    input_image = cv2.imread(input_image_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
    in_width, in_height, in_channels = input_image.shape
    output_image = numpy.zeros((in_width * resize_factor, in_height * resize_factor, in_channels))
    interp_bilinear(input_image, output_image)


if __name__ == "__main__":
    __test_interp()
