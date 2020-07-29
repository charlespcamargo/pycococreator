#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np

import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
import glob


class PyCocoCreator():

    def main(self, args):

        print(args)

        self.DATABASE_NAME = args.database_name
        self.base_path = args.base_path
        self.images_path = args.images_path
        self.masks_path = args.masks_path

        # self.ROOT_DIR = os.path.join('datasets', self.DATABASE_NAME)
        # self.ROOT_DIR, "train/images")
        # self.ROOT_DIR, "train/annotations")

        self.IMAGE_DIR = os.path.join(self.base_path, self.images_path)
        self.ANNOTATION_DIR = os.path.join(self.base_path, self.masks_path)

        self.INFO = {
            "description": self.DATABASE_NAME,
            "url": "https://github.com/waspinator/pycococreator",
            "version": "1.0.0",
            "year": datetime.date.today().year,
            "contributor": "Charles Camargo",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        self.LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-Charles-License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

        self.CATEGORIES = [
            {
                "supercategory": "vegetation",
                "id": 1,
                "name": "hedychium_coronarium"
            }
        ]

        self.coco_output = {
            "info": self.INFO,
            "licenses": self.LICENSES,
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": []
        }

        image_id = 1
        segmentation_id = 1

        # filter for jpeg images
        for root, _, files in os.walk(self.IMAGE_DIR):
            image_files = self.filter_for_images(root, files)
            has_annotation = False

            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                image_info = self.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)

                # filter for associated png annotations
                for root, _, files in os.walk(self.ANNOTATION_DIR):
                    annotation_files = self.filter_for_annotations(
                        root, files, image_filename)

                    if(not annotation_files or len(annotation_files) == 0):
                        print(
                            f'\n-------------------- without annotations_files {image_filename}\n')

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        print(annotation_filename)
                        # [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                        class_id = 1

                        category_info = {'id': class_id,
                                         'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename)
                                                 .convert('1')).astype(np.uint8)

                        self.annotation_info = self.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                        if self.annotation_info is not None:
                            self.coco_output["annotations"].append(
                                self.annotation_info)
                            has_annotation = True

                        segmentation_id = segmentation_id + 1

                if (has_annotation == True):
                    self.coco_output["images"].append(image_info)
                else:
                    print(
                        f'\n------------ The image {image_filename} has no annotations. ------------\n')

                image_id = image_id + 1

        with open(f'{self.base_path}/{self.DATABASE_NAME}.json', 'w') as output_json_file:
            json.dump(self.coco_output, output_json_file)
            print(f"\nfile saved {self.base_path}{self.DATABASE_NAME}.json\n")

    def filter_for_images(self, root, files):
        file_types = ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG', '*.png', '*.PNG']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        return files

    def filter_for_annotations(self, root, files, image_filename):
        file_types = ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG', '*.png', '*.PNG']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(
            os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + '.*'
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [f for f in files if re.match(
            file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

        return files

    def convert(self, text): return int(
        text) if text.isdigit() else text.lower()

    def natrual_key(self, key): return [convert(c)
                                        for c in re.split('([0-9]+)', key)]

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

        if category_info["is_crowd"]:
            is_crowd = 1
            segmentation = binary_mask_to_rle(binary_mask)
        else:
            is_crowd = 0
            segmentation = self.binary_mask_to_polygon(binary_mask, tolerance)
            if not segmentation:
                return None

        self.annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": is_crowd,
            "area": area.tolist(),
            "bbox": bounding_box.tolist(),
            "bbox_mode": 0,  # from detectron2.structures import BoxMode 'BoxMode.XYXY_ABS'
            "segmentation": segmentation,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }

        return self.annotation_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate")

    parser.add_argument("-dn", "--database_name", dest="database_name",
                        default="hedychium_coronarium", help="path to root of datasets")

    parser.add_argument("-b", "--base_path", dest="base_path",
                        default="../images/train/", help="base path to images")

    parser.add_argument("-i", "--images_path", dest="images_path",
                        default="images/", help="path to images")

    parser.add_argument("-m", "--masks_path", dest="masks_path",
                        default="masks/", help="path to masks")

    args = parser.parse_args()

    coco = PyCocoCreator()
    coco.main(args)
