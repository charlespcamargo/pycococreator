#!/usr/bin/env python3

import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
 

class PyCocoCreatorTools():

    def convert(self, text):
        return int(text) if text.isdigit() else text.lower()

    def natrual_key(self, key):
        return [self.convert(c) for c in re.split('([0-9]+)', key)]

    def resize_binary_mask(self, array, new_size):
        image = Image.fromarray(array.astype(np.uint8)*255)
        image = image.resize(new_size)
        return np.asarray(image).astype(np.bool_)

    def close_contour(self, contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    def binary_mask_to_rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))

        return rle

    def binary_mask_to_polygon(self, binary_mask, tolerance=0):
        """Converts a binary mask to COCO polygon representation

        Args:
            binary_mask: a 2D binary numpy array where '1's represent the object
            tolerance: Maximum distance from original points of polygon to approximated
                polygonal chain. If tolerance is 0, the original coordinate array is returned.

        """
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(
            binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:
            contour = self.close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        return polygons

    def create_image_info(self, image_id, file_name, image_size,
                          date_captured=datetime.datetime.utcnow().isoformat(' '),
                          license_id=1, coco_url="", flickr_url=""):

        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
        }

        return image_info

    def create_annotation_info(self, annotation_id, image_id, category_info, binary_mask,
                               image_size=None, tolerance=2, bounding_box=None):

        if image_size is not None:
            binary_mask = self.resize_binary_mask(binary_mask, image_size)

        binary_mask_encoded = mask.encode(
            np.asfortranarray(binary_mask.astype(np.uint8)))

        area = mask.area(binary_mask_encoded)
        if area < 1:
            return None

        if bounding_box is None:
            bounding_box = mask.toBbox(binary_mask_encoded)

        if category_info["is_crowd"] == 1:
            is_crowd = 1
            segmentation = self.binary_mask_to_rle(binary_mask)
        else:
            is_crowd = 0
            segmentation = self.binary_mask_to_polygon(binary_mask, tolerance)
            if not segmentation:
                return None

        # Detectron2 - bbox_mode
        # https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BoxMode
        # detectron2.structures.BoxMode
        # XYXY_ABS = 0
        # XYWH_ABS = 1
        # XYXY_REL = 2
        # XYWH_REL = 3
        # XYWHA_ABS = 4

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": is_crowd,
            "area": area.tolist(),
            "bbox": bounding_box.tolist(),
            "bbox_mode": 0, # 0
            "segmentation": segmentation,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }

        return annotation_info
