from PIL import Image
import math
import numpy as np
import scipy.ndimage
from numpy import sin, cos
import time
import json


def show(img):
    print(np.max(img))
    img_show = img / np.max(img) * 255.0
    pil = Image.fromarray(img_show)
    pil.show()


def denormalize(x):
    limit = 1e-15
    return np.where(np.abs(x) > limit, x, 0.0)


def rotate(img, theta, height):
    c_in = 0.5 * np.array(img.shape)
    c_out = 0.5 * np.array(img.shape)
    a = np.pi * (2.0 * theta / float(height))
    transform = np.array([[np.cos(a), -np.sin(a)],
                          [np.sin(a),  np.cos(a)]])
    transform = denormalize(transform)
    center = 0.5 * np.array(img.shape)
    offset = np.array([0.5, 0.5])
    vector = center - offset
    shift = np.floor(center - vector.dot(transform))
    return scipy.ndimage.affine_transform(img,
                                          transform.T,
                                          order=2,
                                          offset=shift,
                                          output_shape=img.shape,
                                          cval=0.0)


def save(img, filename):
    # img_show = img / np.max(img) * 255.0
    img_show = img
    pil = Image.fromarray(img_show)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil.save(filename)


def reverse_projection(sino, shape):
    img = np.zeros(shape, dtype=float)
    img_width = shape[1]
    img_height = shape[0]
    sino_height = sino.shape[0]
    # stack direction
    # for theta in range(height):
    #  img += np.hstack(sino[:, theta:theta+1] for i in range(width))
    for theta in range(sino_height):
        # img += np.vstack(
        # sino[theta:theta+1, :] for i in range(img_height))
        # img = rotate(img, 1, sino_height)
        stack = np.vstack(sino[theta:theta+1, :] for i in range(img_height))
        img += rotate(stack, -theta, sino_height)
    img = img / img_width
    return img


def nearlest(img, position):
    return img[int(position[0, 0]), int(position[1, 0])]


def liner(img, position):
    pass


def rotate_sum(img, theta):
    # img[y, x]
    center = np.array(img.shape).reshape(2, 1) / 2.0
    width = img.shape[1]
    height = img.shape[0]
    max_len = np.sqrt(width**2 + height**2)
    a = np.pi * 2.0 * (float(theta) / height)
    rotation = np.array([[cos(a), -sin(a)],
                         [sin(a),  cos(a)]]).T
    rotation = denormalize(rotation)

    unit_vector = np.array([[1],
                            [0]])
    unit_vector = np.dot(rotation, unit_vector)
    # print("unit_vector")
    # print(unit_vector)

    ret = np.zeros(width)
    for x in range(width):
        start = np.array((0.5, x + 0.5)).reshape(2, 1) - center
        start = np.dot(rotation, start) + center
        position = start
        inside = False
        count = 0
        while count < (max_len):
            if 0 <= position[0, 0] and position[0, 0] < height and \
               0 <= position[1, 0] and position[1, 0] < width:
                # print(x, position, nearlest(img, position))
                ret[x] += nearlest(img, position)
                inside = True
            else:
                if inside:
                    break
            position += unit_vector
            count += 1
    return ret


def forward_projection(img, shape):
    sino = np.zeros(shape, dtype=float)
    height = sino.shape[0]
    for theta in range(height):
        # rotate_sum(img, theta)
        rotated = rotate(img, theta, height)
        # stack direction
        sino[theta, :] = np.sum(rotated, axis=0)
        # sino[:, theta] = np.sum(rotated, axis=1)
    return sino


shift = 0


def wrap_foward_projection(x, limit=255.0):
    x = x.reshape(shape)
    return np.sum(np.abs(forward_projection(x, sino.shape) - sino)) + shift


def wrap_new(x, limit=255.0):
    ret = wrap_foward_projection(x)
    ret += np.sum(x.reshape(shape) * rev)
    return ret


def wrap_laplace(x, limit=255.0):
    ret = wrap_new(x)
    lap = scipy.ndimage.filters.laplace(x.reshape(shape))
    ret += np.sum(lap)
    return ret


def wrap_drop(x, limit=255.0):
    x = x.reshape(shape)
    p = 0.2
    drop = np.random.binomial(size=shape, n=1, p=1-p)
    x = x * drop
    return np.sum(np.abs(forward_projection(x, sino.shape) - sino)) + shift


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0
    return c


def wrap_double(x, limit=255.0):
    x = x.reshape(shape)
    y_k = forward_projection(x, sino.shape)
    y_i = div0(sino, y_k)
    ramda = reverse_projection(y_i, shape)
    x = x * ramda
    return wrap_foward_projection(x)


def MLEM(img, iteration=100):
    show(img)
    print("making sinogram")
    sino = forward_projection(img, img.shape)
    show(sino)

    print("converting (%d, %d) to (%d, %d)" % (sino.shape + img.shape))
    img = np.ones(img.shape, dtype=float)
    results = []
    start = time.time()
    for i in range(iteration):
        print("iter:", i)
        f_start = time.time()
        y_k = forward_projection(img, img.shape)
        print("forward_projection:", time.time() - start)
        y_i = div0(sino, y_k)
        results.append(np.sum(sino - y_k))
        ramda = reverse_projection(y_i, y_i.shape)
        img = img * ramda
        show(img)
        print("Fitness:", results[-1])
    duration = time.time() - start
    print(duration)
    results = {"time": str(duration), "results": results}
    f = open("result/mlem.json", "w")
    json.dump(results, f)
    f.close()
    show(img)
    save(img, "result/mlem.png")


if __name__ == '__main__':
    ref = Image.open('./test/img/Head-480x480.png')
    ref = ref.convert('L')
    numpy_ref = np.array(ref, dtype=float)
    numpy_ref.flags.writeable = True
    MLEM(numpy_ref)
