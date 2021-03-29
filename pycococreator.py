
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np

class PyCocoCreator():

    def main(self, args, creator_tools):

        # print(args)

        self.DATABASE_NAME = args.database_name
        self.base_path = args.base_path
        self.images_path = args.images_path
        self.masks_path = args.masks_path

        self.IMAGE_DIR = os.path.join(self.base_path, self.images_path)
        self.ANNOTATION_DIR = os.path.join(self.base_path, self.masks_path)

        self.iscrowd = 0

        self.init_file()

        # filter for jpeg images
        for root, _, files in os.walk(self.IMAGE_DIR):
            image_files = self.filter_for_images(root, files)
            has_annotation = False
            self.process_images(image_files, creator_tools)

        self.write_file()

    def init_file(self):
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

    def process_images(self, image_files, creator_tools):
        image_id = 1
        segmentation_id = 1

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = creator_tools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            
            has_annotation = False

            # filter for associated png annotations
            for root, _, files in os.walk(self.ANNOTATION_DIR):
                annotation_files = self.filter_for_annotations(
                    root, files, image_filename)

                if(not annotation_files or len(annotation_files) == 0):
                    print(
                        f'\n-------------------- without annotations_files {image_filename}\n')

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(f'image_id: {image_id} - {annotation_filename}')
                    [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                    class_id = 0

                    category_info = {'id': class_id, 'is_crowd': self.iscrowd}
                    
                    binary_mask = np.asarray(Image.open(
                        annotation_filename).convert('1')).astype(np.uint8)

                    self.annotation_info = creator_tools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask, image.size, tolerance=2)

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

    def write_file(self):
        with open(f'{self.base_path}{self.DATABASE_NAME}/{self.DATABASE_NAME}.json', 'w+') as output_json_file:
            json.dump(self.coco_output, output_json_file)
            
            print(f"\nPyCocoCreator - file saved {self.base_path}{self.DATABASE_NAME}/{self.DATABASE_NAME}.json\n")

    def filter_for_images(self, root, files):
        file_types = ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG', '*.png', '*.PNG']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        ############            
        #files = ['../images/train/images/DJI_0594.JPG']    
        ############

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

    def convert(self, text):
        return int(text) if text.isdigit() else text.lower()

    def natrual_key(self, key):
        return [self.convert(c) for c in re.split('([0-9]+)', key)]
 