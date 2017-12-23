import cv2
import time
import numba
import numpy

from logging import metalogger

log = metalogger.MetaLogger()


@numba.jit("void(u1[:,:,:],u1[:,:,:])", nopython=True, cache=True, nogil=True)
def __numba_interp_bilinear(input_image, output_image):
    height, width = output_image.shape[:-1]
    in_height, in_width = input_image.shape[:-1]

    w_ratio = width / in_width
    h_ratio = height / in_height

    for y in range(height):
        orig_y = float(y) / h_ratio

        i_y = int(orig_y)
        floor_y = orig_y - i_y
        ceil_y = 1 - floor_y
        for x in range(width):
            r = 0.0
            g = 0.0
            b = 0.0

            orig_x = float(x) / w_ratio

            i_x = int(orig_x)
            floor_x = orig_x - i_x
            ceil_x = 1 - floor_x

            mult1 = ceil_x * ceil_y
            mult2 = ceil_x * floor_y
            mult3 = floor_x * ceil_y
            mult4 = floor_x * floor_y

            px1 = input_image[i_y][i_x]
            px2 = input_image[i_y][i_x + 1]
            px3 = input_image[i_y + 1][i_x]
            px4 = input_image[i_y + 1][i_x + 1]

            r += px1[0] * mult1 + px2[0] * mult2
            g += px1[1] * mult1 + px2[1] * mult2
            b += px1[2] * mult1 + px2[2] * mult2

            r += px3[0] * mult3 + px4[0] * mult4
            g += px3[1] * mult3 + px4[1] * mult4
            b += px3[2] * mult3 + px4[2] * mult4

            output_image[y][x][0] = r
            output_image[y][x][1] = g
            output_image[y][x][2] = b


def __python_interp_bilinear(input_image, output_image):
    height, width = output_image.shape[:-1]
    in_height, in_width = input_image.shape[:-1]

    w_ratio = width / in_width
    h_ratio = height / in_height

    for y in range(1, height):
        orig_y = float(y) / h_ratio - 1

        i_y = int(orig_y)
        floor_y = orig_y - i_y
        ceil_y = 1 - floor_y
        for x in range(1, width):

            r = 0.0
            g = 0.0
            b = 0.0

            orig_x = float(x) / w_ratio - 1

            i_x = int(orig_x)
            floor_x = orig_x - i_x
            ceil_x = 1 - floor_x

            mult1 = ceil_x * ceil_y
            mult2 = ceil_x * floor_y
            mult3 = floor_x * ceil_y
            mult4 = floor_x * floor_y

            px1 = input_image[i_y][i_x]
            px2 = input_image[i_y][i_x + 1]
            px3 = input_image[i_y + 1][i_x]
            px4 = input_image[i_y + 1][i_x + 1]

            r += px1[0] * mult1 + px2[0] * mult2
            g += px1[1] * mult1 + px2[1] * mult2
            b += px1[2] * mult1 + px2[2] * mult2

            r += px3[0] * mult3 + px4[0] * mult4
            g += px3[1] * mult3 + px4[1] * mult4
            b += px3[2] * mult3 + px4[2] * mult4

            output_image[y][x][0] = r
            output_image[y][x][1] = g
            output_image[y][x][2] = b


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

    start_time = time.time()

    log.info("Starting up...")

    input_image_path = r"D:\Dev\NumbaLib\Assets\Peter.jpg"
    resize_factor = 10

    input_image = cv2.imread(input_image_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
    in_width, in_height, in_channels = input_image.shape
    output_image = numpy.zeros((in_width * resize_factor, in_height * resize_factor, in_channels), dtype=numpy.uint8)

    # time_before_python_run = time.time()
    # __python_interp_bilinear(input_image, output_image)
    # log.info("Pure Python Version Took %.3f seconds" % (time.time() - time_before_python_run))

    time_before_numba = time.time()
    interp_bilinear(input_image, output_image)
    log.info("Numba Version Took %.3f seconds" % (time.time() - time_before_numba))

    cv2.imwrite(r"D:\Dev\NumbaLib\Assets\Peter_Resized.jpg", output_image)

    log.info("Test interp done @ %.3f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    __test_interp()
