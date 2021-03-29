

# Just For Test in Jupyter/Colab
class Args():
    
    def __init__(self): 
        self.main()

    def main(self,
             stage='test', 
             database_name = 'hedychium_coronarium', 
             images_path = "images/", 
             masks_path = "masks/",
             mask_definition = 'mask_definition.json',
             instances_json = 'coco_instances.json',
             max_width = 920,
             image_id = None,
             width = 4000,
             height = 3000,
             width_to_resize = 4000,
             height_to_resize = 3000,
             generate_automatic_info = False,
             iscrowd = 1):
        # stage = train, test, val
        self.stage= stage
        self.database_name = database_name
        self.images_path = images_path
        self.masks_path = masks_path
        self.mask_definition = mask_definition
        self.instances_json = instances_json
        self.max_width = max_width
        self.image_id = image_id
        self.width = width
        self.height = height
        self.width_to_resize = width_to_resize
        self.height_to_resize = height_to_resize
        self.generate_automatic_info = generate_automatic_info
        self.iscrowd = iscrowd

        self.base_path = f'../{self.database_name}/{self.stage}/'
        self.annotation_path = f'../{self.database_name}/{self.stage}/coco_instances.json'
        # coco_json_utils
        self.dataset_info = f'../{self.database_name}/{self.stage}/dataset_info.json'
    
    def show(self):
        print(f'stage: {self.stage}')
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
        print(f'iscrowd: {self.iscrowd}')

