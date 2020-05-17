import attr
from . import models, utils

import tensorflow as tf
import numpy as np
from PIL import Image

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@attr.s
class Processor:
    _model = attr.ib(init=False, default=None)
    _model_type = attr.ib(default="resnet50")
    _stride = attr.ib(default=16)
    _quant_bytes = attr.ib(default=1)
    _multiplier = attr.ib(default=1.0)

    def __attrs_post_init__(self):
        """
        Load model based on initialization parameters
        """
        self._model = models.load_model(
            self._model_type, self._stride, self._quant_bytes, self._multiplier
        )

    def _preprocess_image(self, image):
        img = tf.keras.preprocessing.image.array_to_img(image)
        width, height = img.size

        targetWidth = (int(width) // self._stride) * (self._stride + 1)
        targetHeight = (int(height) // self._stride) * (self._stride + 1)

        img = img.resize((targetWidth, targetHeight))
        x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)

        # Run neural net specific preprocessing
        # For Resnet
        if "resnet" in self._model_type:
            # add imagenet mean - extracted from body-pix source
            m = np.array([-123.15, -115.90, -103.06])
            x = np.add(x, m)
        # For Mobilenet
        elif "mobilenet" in self._model_type:
            x = (x / 127.5) - 1
        else:
            print("Unknown Model")
        x = x[tf.newaxis, ...]
        return x

    def process_image(self, image):
        img = self._preprocess_image(image)
        # Get input and output tensors
        input_tensor_names = utils.get_input_tensors(self._model)
        logger.debug(input_tensor_names)
        output_tensor_names = utils.get_output_tensors(self._model)
        logger.debug(output_tensor_names)
        input_tensor = self._model.get_tensor_by_name(input_tensor_names[0])

        with tf.compat.v1.Session(graph=self._model) as sess:
            results = sess.run(output_tensor_names, feed_dict={input_tensor: img})
            logger.debug(
                "done. {} outputs received".format(len(results))
            )  # should be 8 outputs

        for idx, name in enumerate(output_tensor_names):
            if "displacement_bwd" in name:
                logger.debug("displacement_bwd %s", results[idx].shape)
            elif "displacement_fwd" in name:
                logger.debug("displacement_fwd %s", results[idx].shape)
            elif "float_heatmaps" in name:
                heatmaps = np.squeeze(results[idx], 0)
                logger.debug("heatmaps %s", heatmaps.shape)
            elif "float_long_offsets" in name:
                longoffsets = np.squeeze(results[idx], 0)
                logger.debug("longoffsets %s", longoffsets.shape)
            elif "float_short_offsets" in name:
                offsets = np.squeeze(results[idx], 0)
                logger.debug("offests %s", offsets.shape)
            elif "float_part_heatmaps" in name:
                partHeatmaps = np.squeeze(results[idx], 0)
                logger.debug("partHeatmaps %s", partHeatmaps.shape)
            elif "float_segments" in name:
                segments = np.squeeze(results[idx], 0)
                logger.debug("segments %s", segments.shape)
            elif "float_part_offsets" in name:
                partOffsets = np.squeeze(results[idx], 0)
                logger.debug("partOffsets %s", partOffsets.shape)
            else:
                logger.debug("Unknown Output Tensor %s %s", name, idx)

        return (heatmaps, longoffsets, offsets, partHeatmaps, segments, partOffsets)

    def evaluate_segmentation(self, segments, image):
        img = tf.keras.preprocessing.image.array_to_img(image)
        width, height = img.size

        targetWidth = (int(width) // self._stride) * (self._stride + 1)
        targetHeight = (int(height) // self._stride) * (self._stride + 1)

        img = img.resize((targetWidth, targetHeight))

        # BODYPART SEGMENTATION
        partOffsetVector = []
        partHeatmapPositions = []
        partPositions = []
        partScores = []
        partMasks = []

        # Segmentation MASk
        segmentation_threshold = 0.7
        segmentScores = tf.sigmoid(segments)
        mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
        logger.debug("maskshape %s", mask.shape)
        segmentationMask = tf.dtypes.cast(mask, tf.uint8)
        segmentationMask = np.reshape(
            segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1])
        )
        logger.debug("maskValue %s", segmentationMask[:][:])

        # segmentationMask_inv = np.bitwise_not(mask_img)
        segmentationMask_inv = np.zeros_like(segmentationMask, dtype=np.uint8)
        segmentationMask_inv[np.nonzero(segmentationMask==0)] = 1
        # Draw Segmented Output
        # Set color to chroma green
        # segmentation_img = np.stack(
        #     (
        #         np.zeros_like(segmentationMask_inv, dtype=np.uint8),
        #         segmentationMask_inv * 177,
        #         segmentationMask_inv * 64,
        #     ),
        #     axis=-1,
        # )
        # print(segmentation_img.shape)
        # print(segmentation_img.dtype)
        # mask_img = Image.fromarray(segmentation_img)
        mask_img = Image.fromarray(segmentationMask * 255, "L")
        mask_img = mask_img.resize((width, height), Image.LANCZOS)

        # fg = np.bitwise_and(np.array(img), np.array(mask_img))
        # bg = np.bitwise_and(np.array(img), np.array(segmentationMask_inv))
        return np.array(mask_img)

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, m_type):
        self._model_type = m_type
        self._model = models.load_model(
            self._model_type, self._stride, self._quant_bytes, self._multiplier
        )

    def target_shape(self, width, height):
        return (
            (int(width) // self._stride) * (self._stride + 1),
            (int(height) // self._stride) * (self._stride + 1),
            3,
        )
