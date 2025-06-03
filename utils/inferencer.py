import os
from mmseg.apis import MMSegInferencer

inferencer = MMSegInferencer(model="work_dirs/mask2former_r50_8xb2-4k_poquets-complet-512x512/best_mioU_iter_1650.pth")

image_dir = "./data/poquets_complet/images"

images = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)[:5]]

inferencer(images, out_dir="output", img_out_dir="vis", pred_out_dir="pred")
