import json
import glob
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
from math import trunc
import base64
from io import BytesIO
import os
import datetime

import IPython


class CocoDataset():
    def main(self, args):

        self.base_path = args.base_path 
        self.annotation_path = os.path.join(self.base_path, args.database_name, args.instances_json)
        self.image_dir = os.path.join(self.base_path, args.images_path)
        self.mask_dir = os.path.join(self.base_path, args.masks_path) 
        self.max_width = args.max_width
        self.image_id = args.image_id

        # Customize these segmentation colors if you like, if there are more segmentations
        # than colors in an image, the remaining segmentations will default to white
        self.colors = ['red', 'green', 'blue', 'yellow']

        if(not os.path.exists(self.annotation_path)): 
            raise Exception(f'File not found {self.annotation_path}, please generate before run.')
        
        with open(self.annotation_path) as json_file:
            self.coco = json.load(json_file)
            json_file.close() 

        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

    def _process_info(self):
        self.info = self.coco['info']

    def _process_licenses(self):
        self.licenses = self.coco['licenses']

    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()

        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')

            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}
                self.super_categories[super_category] |= {cat_id}

    def _process_images(self):

        total_image = len(glob.glob(self.image_dir + '/*'))
        total_masks = len(glob.glob(self.mask_dir + '/*'))

        print(f"Images total: {total_image}")
        print(f"Masks total: {total_masks}\n")

        if(total_image != total_masks):
            raise Exception(
                f'sorry, there is not segmentations for image_id...')

        self.images = dict()
        self.annotations = dict()

        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
                self.annotations[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')


    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def display_info(self):
        print('Dataset Info')
        print('==================')
        for key, item in self.info.items():
            print(f'  {key}: {item}')

    def display_licenses(self):
        print('Licenses')
        print('==================')
        for license in self.licenses:
            for key, item in license.items():
                print(f'  {key}: {item}')

    def display_categories(self):
        print('Categories')
        print('==================')
        for sc_name, set_of_cat_ids in self.super_categories.items():
            print(f'  super_category: {sc_name}')
            for cat_id in set_of_cat_ids:
                print(f'    id {cat_id}: {self.categories[cat_id]["name"]}')

    def load_image(self, img_dir, image):
        # Open the image
        image_path = Path(img_dir) / image['file_name']
        image = PILImage.open(image_path)

        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        data_uri = base64.b64encode(buffer.read()).decode('ascii')
        return image, "data:image/png;base64,{0}".format(data_uri)

    def resize_image(self, image):
        max_width = self.max_width
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height

        return adjusted_width, adjusted_ratio, adjusted_height

    def save_images_to_html(self, images_ids, max_width=880, show_bbox=True, show_polys=True, show_crowds=True):

        html = None

        print('Images')
        print('==================')

        data = datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")

        html_file = open(f"results/{data}_00.html", "w")
        css = self.get_css()

        for image_id in images_ids:
            html = self.save_image_to_html(
                image_id=image_id, max_width=max_width)
            html_file.write(html)

            if (image_id > 0 and image_id % 5 == 0):
                html_file.write(css)
                html_file.close()
                print(f"\nfile saved at: results/{data}_{image_id}.html\n")
                print('==================')

                html_file = open(f"results/{data}_{image_id}.html", "w")

        if (html != None and (len(images_ids) == 1 or image_id % 5 != 0)):
            html_file.write(css)
            html_file.close()
            print(f"\nfile saved at: results/{data}_{image_id}.html\n")
            print('==================')

    def save_image_to_html(self, image_index=None, image_id=None, max_width=880, show_bbox=True, show_polys=True, show_crowds=True):

        html = ""
        print('==================')

        if (image_index is None and image_id is None):
            raise ValueError("You need to pass [image_index] or [image_id]")

        if (image_index and image_id):
            raise ValueError(
                "You need to pass only one parameter, choose wisely between: [image_index] and [image_id]")

        if(image_index is not None):
            image_id = list(self.images)[image_index]

        # Print image info
        image = self.images[image_id]
        mask_image = self.annotations[image_id]

        for key, val in image.items():
            print(f'  {key}: {val}')

        # Open the image
        image, image_path = self.load_image(self.image_dir, image)
        mask_image, mask_path = self.load_image(self.mask_dir, mask_image)

        if(max_width != None and max_width > 0):
            self.max_width = max_width

        # Calculate the size and adjusted display size
        adjusted_width, adjusted_ratio, adjusted_height = self.resize_image(
            image)
        adj_width_mask, adj_ratio_mask, adj_height_mask = self.resize_image(
            mask_image)

        print(f'ImageId: {image_id} - RESIZE: w: {adj_width_mask} - h: {adj_height_mask} - ratio: {adjusted_ratio}')

        # Create bounding boxes and polygons
        bboxes = dict()
        polygons = dict()
        rle_regions = dict()
        seg_colors = dict()

        if(not self.segmentations.get(image_id)):
            print(f'segmentation not found! {image_id}')

        else:
            for i, seg in enumerate(self.segmentations[image_id]):
                if i < len(self.colors):
                    seg_colors[seg['id']] = self.colors[i]
                else:
                    seg_colors[seg['id']] = 'white'

                print(
                    f'  {seg_colors[seg["id"]]}: {self.categories[seg["category_id"]]["name"]}')

                bboxes[seg['id']] = np.multiply(
                    seg['bbox'], adjusted_ratio).astype(int)

                # meus dados estão errados, por algum motivo, ele é 1, mas está igual a zero
                if seg['iscrowd'] == 1:
                    polygons[seg['id']] = []
                    for seg_points in seg['segmentation']:
                        seg_points = np.multiply(
                            seg_points, adjusted_ratio).astype(int)
                        polygons[seg['id']].append(
                            str(seg_points).lstrip('[').rstrip(']'))
                else:
                    # Decode the RLE
                    px = 0
                    rle_list = []

                    for j, counts in enumerate(seg['segmentation'][0]):

                        if counts < 0:
                            print(
                                f'ERROR: One of the counts was negative, treating as 0: {counts}')
                            counts = 0

                        if j % 2 == 0:
                            # Empty pixels
                            px += counts
                        else:
                            # Create one or more vertical rectangles
                            x1 = trunc(px / adjusted_height)
                            y1 = px % adjusted_height
                            px += counts
                            x2 = trunc(px / adjusted_height)
                            y2 = px % adjusted_height

                            if x2 == x1:  # One vertical column
                                line = [x1, y1, 1, (y2 - y1)]
                                line = np.multiply(line, adjusted_ratio)
                                rle_list.append(line)
                            else:  # Two or more columns
                                # Insert left-most line first
                                left_line = [x1, y1, 1, (adjusted_height - y1)]
                                left_line = np.multiply(
                                    left_line, adjusted_ratio)
                                rle_list.append(left_line)

                                # Insert middle lines (if needed)
                                lines_spanned = x2 - x1 + 1
                                if lines_spanned > 2:  # Two columns won't have a middle
                                    middle_lines = [
                                        (x1 + 1), 0, lines_spanned - 2, adjusted_height]
                                    middle_lines = np.multiply(
                                        middle_lines, adjusted_ratio)
                                    rle_list.append(middle_lines)

                                # Insert right-most line
                                right_line = [x2, 0, 1, y2]
                                right_line = np.multiply(
                                    right_line, adjusted_ratio)
                                rle_list.append(right_line)

                    if len(rle_list) > 0:
                        rle_regions[seg['id']] = rle_list

        html = self.create_html(image_id, image_path, mask_path, adjusted_width, adjusted_height,
                                show_polys, polygons, show_crowds, rle_regions, seg_colors, show_bbox, bboxes, html)

        return html

    def create_html(self, image_id, image_path, mask_path, adjusted_width, adjusted_height, show_polys, polygons, show_crowds, rle_regions, seg_colors, show_bbox, bboxes, html):

        segs = 0
        if(len(bboxes) > 0):
            segs = len(self.segmentations[image_id])

        # Draw the image
        html += f'<div>'
        html += f'<b>Image: </b>{image_id} - <b>Segmentations:</b> {segs} <br />'
        html += f'<div class="container" >'        
        html += f'<img src="{str(image_path)}" style="width:{adjusted_width}px;">'
        html += f'<svg >'

        # Draw shapes on image
        if show_polys:
            for seg_id, points_list in polygons.items():
                for points in points_list:
                    html += f'<polygon points="{points}" \
                        style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; fill-opacity:0.5; stroke-width:1;" />'

        if show_crowds:
            for seg_id, line_list in rle_regions.items():
                for line in line_list:
                    html += f'<rect x="{line[0]}" y="{line[1]}" width="{line[2]}" height="{line[3]}" \
                        style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; \
                        fill-opacity:0.5; stroke-opacity:0.5" />'

        if show_bbox:
            for seg_id, bbox in bboxes.items():
                html += f'<rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]}" height="{bbox[3]}" \
                    style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; fill-opacity:0.5" />'

        html += '</svg>'
        html += '</div>'

        # Draw the mask image
        html += f'<div class="container" style="opacity: 0.15" >'
        html += f'<img src="{str(mask_path)}" style="width:{adjusted_width}px;">'
        html += '</div>'
        html += '</div>'

        return html

    def get_css(self):
        css = "<style>"
        css += " .container { position: relative; display: inline-block; transition: transform 150ms ease-in-out; } "
        css += " .container img {  display: block; max-width: 100%;height:auto; }"
        css += " .container svg { position: absolute;top: 0px;left: 0; width:100%; height:100%; }"
        css += "</style>"

        return css


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate")

    parser.add_argument("-j", "--instances_json", dest="instances_json", default="coco_instances.json",
                        help="path to JSON path of coco instances")

    parser.add_argument("-i", "--images_path", dest="images_path",
                        default="images/", help="path to images")

    parser.add_argument("-m", "--masks_path", dest="masks_path",
                        default="masks/", help="path to masks")

    parser.add_argument("-mw", "--max_width", dest="max_width", default=920, type=int,
                        help="max width to show images")

    parser.add_argument("-id", "--image_id", dest="image_id", default=10, type=int,
                        help="image to open/generate HTML")

    parser.add_argument("-dn", "--database_name", dest="database_name",
                        default="hedychium_coronarium", help="path to root of datasets")
    parser.add_argument("-b", "--base_path", dest="base_path",
                        default="../images/train/", help="base path to images")

    #args = parser.parse_args()

    #ds = CocoDataset()
    #ds.main(args)

    # all loaded images
    #images_ids = ds.images

    # take just some of all
    # n = 20
    # images_ids = list(images_ids)[0:n]
    # ds.display_categories()
    # ds.save_images_to_html(images_ids, max_width=440)
