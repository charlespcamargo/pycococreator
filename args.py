

# Just For Test in Jupyter/Colab
class Args():
    database_name = 'hedychium_coronarium'
    base_path = '../images/train/'
    images_path = "images/"
    masks_path = "masks/"
    mask_definition = 'mask_definitions.json'
    instances_json = 'coco_instances.json'
    annotation_path = '../images/train/hedychium_coronarium/coco_instances.json'
    max_width = 920
    image_id = 10    
    
    # coco_json_utils
    dataset_info = '../images/train/hedychium_coronarium/dataset_info.json'
    generate_automatic_info = False
    width_to_resize = 4000
    height_to_resize = 3000

    width = 4000
    height = 3000

    def show(self):
        print(f'database_name: {self.database_name}')
        print(f'base_path: {self.base_path}')
        print(f'image_path: {self.images_path}')
        print(f'masks_path: {self.masks_path}')
        print(f'mask_definition: {self.mask_definition}')
        print(f'instances_json: {self.instances_json}')
        print(f'max_width: {self.max_width}')
        print(f'image_id: {self.image_id}')
        print(f'dataset_info: {self.dataset_info}')
        print(f'generate_automatic_info: {self.generate_automatic_info}')
        print(f'width_to_resize: {self.width_to_resize}')
        print(f'height_to_resize: {self.height_to_resize}')
        print(f'width: {self.width}')
        print(f'height: {self.height}')

