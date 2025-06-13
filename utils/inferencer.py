import os
from mmseg.apis import MMSegInferencer

inferencer = MMSegInferencer(model="work_dirs/mask2former-pipeline1/mask2former_swin-l-in22k-384x384-pre_8xb2-10k_poquets-complets-640x640.py")

image_dir = "./data/poquets_complet/images/test"

images = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)[:5]]

inferencer(images, out_dir="output", img_out_dir="vis", pred_out_dir="pred")
