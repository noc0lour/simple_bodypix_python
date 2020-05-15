import attr
from . import models, utils

import tensorflow as tf
import numpy as np

import logging

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

    def process_image(self, image):
        self._mode
        # Get input and output tensors
        input_tensor_names = utils.get_input_tensors(self._model)
        logger.debug(input_tensor_names)
        output_tensor_names = utils.get_output_tensors(self._model)
        logger.debug(output_tensor_names)
        input_tensor = self._model.get_tensor_by_name(input_tensor_names[0])

        with tf.compat.v1.Session(graph=self._model) as sess:
            results = sess.run(output_tensor_names, feed_dict={input_tensor: image})
            logger.debug(
                "done. {} outputs received".format(len(results))
            )  # should be 8 outputs

        for idx, name in enumerate(output_tensor_names):
            if "displacement_bwd" in name:
                logger.debug("displacement_bwd %s", results[idx].shape)
            elif "displacement_fwd" in name:
                logger.debug("displacement_fwd %?", results[idx].shape)
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
