import json
import glob
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
from math import trunc
import base64
from io import BytesIO

import IPython


class CocoDataset():
    def main(self, args):

        self.annotation_path = args.instances_json_path
        self.image_dir = args.images_path
        self.mask_dir = args.masks_path
        self.max_width = args.max_width
        self.image_id = args.image_id

        # Customize these segmentation colors if you like, if there are more segmentations
        # than colors in an image, the remaining segmentations will default to white
        self.colors = ['red', 'green', 'blue', 'yellow']

        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

        html = self.display_image(image_id=self.image_id, max_width=self.max_width)

        with open("file.html", "w") as file:
            file.write(html)

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

        print(f"Images total: {total_image}\n")
        print(f"Masks total: {total_masks}\n")

        if(total_image != total_masks):
            raise f'sorry, there is not segmentations for image_id: {image_id}'

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
                print(f'    id {cat_id}: {self.categories[cat_id]["name"]}'
                      )

            print('')

    def display_image(self, image_id, show_bbox=True, show_polys=True, show_crowds=True, max_width=4000):
        print('Image')
        print('==================')

        # Print image info
        image = self.images[image_id]
        mask = self.annotations[image_id]

        for key, val in image.items():
            print(f'  {key}: {val}')

        file_name = image['file_name']
        # Open the image
        image, image_path = self.load_image(image)
        mask, mask_path = self.load_image(mask, file_name)

        # Calculate the size and adjusted display size
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height

        # Create bounding boxes and polygons
        bboxes = dict()
        polygons = dict()
        rle_regions = dict()
        seg_colors = dict()

        try:
            self.segmentations[image_id]
        except IndexError:
            raise f'sorry, there is not segmentations for image_id: {image_id}'

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
            # if seg['iscrowd'] == 0 or seg['iscrowd'] == 1:
            if seg['iscrowd'] == 0:
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

                for j, counts in enumerate(seg['segmentation']['counts']):
                    if counts < 0:
                        print(
                            f'ERROR: One of the counts was negative, treating as 0: {counts}')
                        counts = 0

                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Create one or more vertical rectangles
                        x1 = trunc(px / image_height)
                        y1 = px % image_height
                        px += counts
                        x2 = trunc(px / image_height)
                        y2 = px % image_height

                        if x2 == x1:  # One vertical column
                            line = [x1, y1, 1, (y2 - y1)]
                            line = np.multiply(line, adjusted_ratio)
                            rle_list.append(line)
                        else:  # Two or more columns
                            # Insert left-most line first
                            left_line = [x1, y1, 1, (image_height - y1)]
                            left_line = np.multiply(left_line, adjusted_ratio)
                            rle_list.append(left_line)

                            # Insert middle lines (if needed)
                            lines_spanned = x2 - x1 + 1
                            if lines_spanned > 2:  # Two columns won't have a middle
                                middle_lines = [
                                    (x1 + 1), 0, lines_spanned - 2, image_height]
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

        html = '<div class="container" style="position:relative;float:left">'
        html += f'<img id="mask_img" src="{str(mask_path)}" style="position:relative; top:0px; left:0px;width:{adjusted_width}">'
        html += '</div>'

        # Draw the image
        html += '<div class="container" style="position:relative;float:right">'
        html += f'<img src="{str(image_path)}" style="position:relative; top:0px; left:0px;width: {adjusted_width}">'
        html += '<div class="svgclass">'
        html += f'<svg width="{adjusted_width}" height="{adjusted_height}">'

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
                    style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; fill-opacity:0" />'

        html += '</svg>'
        html += '</div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass {position: absolute; top:0px; left: 0px}'
        html += '</style>'

        return html

    def load_image(self, image, file_name=''):
        image_path = None

        if(not file_name):
            image_path = Path(self.image_dir) / image['file_name']
        else:
            image_path = Path(self.mask_dir) / file_name

        image = PILImage.open(image_path)

        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        data_uri = base64.b64encode(buffer.read()).decode('ascii')
        image_path = "data:image/png;base64,{0}".format(data_uri)

        return image, image_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate")

    parser.add_argument("-j", "--instances_json_path", dest="instances_json_path", default="datasets/hedychium_coronarium/hedychium_coronarium.json",
                        help="path to JSON path of coco instances")

    parser.add_argument("-i", "--images_path", dest="images_path",
                        default="datasets/hedychium_coronarium/train/images", help="path to images")

    parser.add_argument("-m", "--masks_path", dest="masks_path",
                        default="datasets/hedychium_coronarium/train/annotations", help="path to masks")

    parser.add_argument("-mw", "--max_width", dest="max_width", default=920, type=int,
                        help="max width to show images")

    parser.add_argument("-id", "--image_id", dest="image_id", default=10, type=int,
                        help="image to open/generate HTML") 

    args = parser.parse_args()

    ds = CocoDataset()
    ds.main(args)
