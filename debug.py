import os
root = 'MasterDataset/train'
print('Root exists:', os.path.exists(root))
plants = os.listdir(root)[:3]
print('Plants:', plants)
for plant in plants:
    plant_path = os.path.join(root, plant)
    print(f'{plant}: isdir={os.path.isdir(plant_path)}, contents={len(os.listdir(plant_path)) if os.path.isdir(plant_path) else 0}')
