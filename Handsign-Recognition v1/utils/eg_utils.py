import cv2

__author__ = 'ADF-AI'


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = 2.0 * (x - 0.5)
    return x


def draw_text(coordinates, img, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(img, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_bounding_box(coordinates, img, color):
    x, y, w, h = coordinates
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def apply_offsets(coordinates, offsets):
    x, y, width, height = coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off

# copyright (c) 2017 - Artificial Intelligence Laboratory, Telkom University #
