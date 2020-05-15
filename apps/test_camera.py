#!/usr/bin/env python3

import acapture
import pyglview
import cv2

import sys


def loop(capture, viewer):
    check, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if check:
        viewer.set_image(frame)



def main():
    viewer = pyglview.Viewer()
    cap = acapture.open(0)
    viewer.set_loop(lambda: loop(cap, viewer))
    viewer.start()


if __name__ == "__main__":
    sys.exit(not main())
