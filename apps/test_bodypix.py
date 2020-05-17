#!/usr/bin/env python3

import acapture
import pyglview
import cv2
import PIL

from bodypix import processor

import sys
import argparse
import logging
import ctypes
import multiprocessing as mp
import numpy as np
import pyfakewebcam
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    return parser.parse_args()


def processing_loop(input_buffer, input_shape, segmentation_buffer, segmentation_shape):
    proc = processor.Processor(quant_bytes=2, stride=32)
    segmentation_array = np.frombuffer(
        segmentation_buffer.get_obj(), dtype=np.uint8
    ).reshape(segmentation_shape)
    input_array = np.frombuffer(input_buffer.get_obj(), dtype=np.uint8).reshape(
        input_shape
    )
    data = np.empty_like(input_array)
    while True:
        # don't let the other process modify while we are copying
        with input_buffer.get_lock():
            np.copyto(data, input_array)

        start = time.perf_counter()
        results = proc.process_image(data)
        end = time.perf_counter()
        print("time: ", end - start)
        segmentation = proc.evaluate_segmentation(results[4], data)
        with segmentation_buffer.get_lock():
            np.copyto(segmentation_array, segmentation)


def viewer_loop(input_buffer, input_shape, segmentation_buffer, segmentation_shape):
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)
    input_array = np.frombuffer(input_buffer.get_obj(), dtype=np.uint8).reshape(
        input_shape
    )
    segmentation_array = np.frombuffer(
        segmentation_buffer.get_obj(), dtype=np.uint8
    ).reshape(segmentation_shape)

    background = PIL.Image.new("RGB", (input_shape[1], input_shape[0]), (0, 177, 64))
    camera = pyfakewebcam.FakeWebcam("/dev/video2", 640, 480)

    def loop():
        check, camera_input = capture.read()
        if check:
            camera_input = cv2.cvtColor(camera_input,cv2.COLOR_BGR2RGB)
            with input_buffer.get_lock():
                np.copyto(input_array, camera_input)
            img = PIL.Image.fromarray(camera_input)
            # Do this async on latest frame only
            with segmentation_buffer.get_lock():
                mixed_img = background.copy()
                mixed_img.paste(img, mask=PIL.Image.fromarray(segmentation_array))

            viewer.set_image(np.array(mixed_img))
            camera.schedule_frame(np.array(mixed_img))

    viewer = pyglview.Viewer()
    viewer.set_loop(loop)
    viewer.start()


def main():
    args = parse_args()
    if args.debug:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # add formatter to ch
        ch.setFormatter(formatter)
        logging.getLogger("").addHandler(ch)
        logging.getLogger("").setLevel(logging.DEBUG)

    cap_dtype = np.uint8
    cap_shape = (480, 640, 3)
    seg_shape = (480, 640)

    input_buffer = mp.Array(
        np.ctypeslib.as_ctypes_type(cap_dtype), int(np.prod(cap_shape))
    )
    segmentation_buffer = mp.Array(
        np.ctypeslib.as_ctypes_type(cap_dtype), int(np.prod(seg_shape))
    )
    np.copyto(
        np.frombuffer(segmentation_buffer.get_obj(), dtype=cap_dtype),
        np.full((int(np.prod(seg_shape))), 255, dtype=cap_dtype),
    )
    procs = []
    # procs.append(
    #     mp.Process(
    #         target=viewer_loop,
    #         args=(cap, input_buffer, cap_shape, segmentation_buffer, output_shape),
    #     )
    # )
    procs.append(
        mp.Process(
            target=processing_loop,
            args=(input_buffer, cap_shape, segmentation_buffer, seg_shape),
        )
    )

    for p in procs:
        p.start()
    viewer_loop(input_buffer, cap_shape, segmentation_buffer, seg_shape)
    for p in procs:
        p.join()


if __name__ == "__main__":
    sys.exit(not main())
