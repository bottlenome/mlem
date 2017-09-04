import numpy


def rotation_matrix(theta):
    return numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                        [numpy.sin(theta), numpy.cos(theta)]])


def calc_system_matrix(shape, detector_num, rotation_num):

    def setVal(mat, i, x, w, x1, y1, shape, val):
        if (0 <= x1 and x1 < shape[0] and 0 <= y1 and y1 < shape[1]):
            x_ = x1 + y1 * shape[0]
            y_ = i * w + x
            mat[y_, x_] += val
            # print("setval", x_, y_, val)

    # TODO remove this limitation
    assert(shape[0] == detector_num)
    ret = numpy.zeros((shape[0] * shape[1], detector_num * rotation_num))
    length = numpy.sqrt(shape[0] ** 2 + shape[1] ** 2)
    # print(length)
    center_x = shape[0] / 2.0
    center_y = shape[1] / 2.0
    center = numpy.array([[center_x], [center_y]])
    # print(center)
    for i in range(rotation_num):
        # print("i", i)
        theta = 2.0 * numpy.pi * i / rotation_num
        for x in range(detector_num):
            # print("x", x)
            v = numpy.dot(rotation_matrix(theta), numpy.array([[0], [1]]))
            # print("v", v)
            to_start = numpy.array([[-center_x + x + 0.5], [-length / 2]])
            # print("to_start", to_start)
            start = center + numpy.dot(rotation_matrix(theta), to_start)
            # print("start", start)
            for y in range(int(length) + 1):
                p = start + v * y
                # print("p", p)
                x_ = p[0] - 0.5
                y_ = p[1] - 0.5
                x1 = int(numpy.floor(x_))
                y1 = int(numpy.floor(y_))
                x2 = x1 + 1
                y2 = y1 + 1
                setVal(ret, i, x, shape[0],
                       x1, y1, shape, (x2 - x_) * (y2 - y_))
                setVal(ret, i, x, shape[0],
                       x2, y1, shape, (x_ - x1) * (y2 - y_))
                setVal(ret, i, x, shape[0],
                       x1, y2, shape, (x2 - x_) * (y_ - y1))
                setVal(ret, i, x, shape[0],
                       x2, y2, shape, (x_ - x1) * (y_ - y1))

    return ret


def border(img, x, y):
    if x >= img.shape[0] or y >= img.shape[1]:
        return 0
    else:
        return img[y, x]


def linear2d(img, x, y):
    ret = 0.0
    x1 = int(numpy.floor(x))
    y1 = int(numpy.floor(y))
    x2 = x1 + 1
    y2 = y1 + 1
    ret += border(img, x1, y1) * ((x2 - x) * (y2 - y))
    ret += border(img, x2, y1) * ((x - x1) * (y2 - y))
    ret += border(img, x1, y2) * ((x2 - x) * (y - y1))
    ret += border(img, x2, y2) * ((x - x1) * (y - y1))
    return ret


if __name__ == '__main__':
    import unittest

    class TestLinear2d(unittest.TestCase):
        a = numpy.array([[1, 2],
                         [3, 4]])

        def test_left(self):
            self.assertEqual(linear2d(self.a, 0, 0), 1)

        def test_left_center(self):
            self.assertEqual(linear2d(self.a, 0.5, 0), 1.5)

        def test_right(self):
            self.assertEqual(linear2d(self.a, 1, 0), 2)

        def test_right_center(self):
            self.assertEqual(linear2d(self.a, 1.5, 0), 1)

        def test_center(self):
            self.assertEqual(linear2d(self.a, 0.5, 0.5), 2.5)

        def test_calc_system_matrix(self):
            s = calc_system_matrix([8, 8], 8, 8)
            numpy.set_printoptions(precision=3, suppress=True)
            # |
            print(s[0].reshape(8, 8))
            # /
            print(s[8].reshape(8, 8))
            print(s[11].reshape(8, 8))
            print(s[12].reshape(8, 8))
            # -
            print(s[16].reshape(8, 8))
            self.assertEqual(s.shape[0], 64)
            self.assertEqual(s.shape[1], 64)

        def test_compare_width_rotate(self):
            s = calc_system_matrix([8, 8], 8, 8)
            img = numpy.random.random((8, 8))
            result_s = numpy.dot(s, img.reshape(img.size, 1))
            result_s = result_s.reshape(8, 8)
            import mlem
            result_m = mlem.forward_projection(img, img.shape)
            numpy.set_printoptions(precision=3, suppress=True)
            print(result_s - result_m)
            print(numpy.sum(result_s - result_m))

        def test_show_sino(self):
            from PIL import Image
            dst = Image.open('./test/img/Head-16x16.png')
            dst = dst.convert('L')
            numpy_dst = numpy.array(dst, dtype=float)
            s = calc_system_matrix([16, 16], 16, 16)
            sino = numpy.dot(s, numpy_dst.reshape(numpy_dst.size, 1))
            sino = sino.reshape(16, 16)
            sino = Image.fromarray(sino / numpy.max(sino) * 255.0).convert('L')
            # sino.save('system_sino.png')

    unittest.main()
