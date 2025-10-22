import os
from data_loader import PlantDataset

root_dir = 'MasterDataset/train'
print('Root dir:', root_dir)
print('Exists:', os.path.exists(root_dir))

ds = PlantDataset(root_dir)
print('Classes:', ds.classes)
print('Samples:', len(ds.samples))

# Debug the loop
for plant_type in os.listdir(root_dir)[:2]:  # First 2
    print(f'Plant type: {plant_type}')
    plant_path = os.path.join(root_dir, plant_type)
    print(f'  Plant path: {plant_path}, isdir: {os.path.isdir(plant_path)}')
    if os.path.isdir(plant_path):
        for condition in os.listdir(plant_path)[:2]:  # First 2 conditions
            print(f'  Condition: {condition}')
            condition_path = os.path.join(plant_path, condition)
            print(f'    Condition path: {condition_path}, isdir: {os.path.isdir(condition_path)}')
            if os.path.isdir(condition_path):
                class_name = f"{plant_type}_{condition}"
                print(f'    Class name: {class_name}')
                img_count = len([f for f in os.listdir(condition_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f'    Images: {img_count}')
