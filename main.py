if __name__ == "__main__":

    # from utils.train import train


    # config_file = "./configs/my_configs/mask2former_r50_8xb2_poquets-complet-512x512.py"

    # train(config_file)

    import os
    from utils.visualization import show_image
    from mmseg.datasets.transforms import transforms
    from mmcv.transforms import RandomChoiceResize, CenterCrop
    from PIL import Image
    import numpy as np
    import time

    pipeline_new = [
        transforms.RandomUniformDownScale(min_down_scale=3.0, max_down_scale=5.0),
        transforms.RandomCrop(crop_size=(640, 640)),
        transforms.RandomFlip(prob=0.5, direction="vertical"),
        transforms.RandomFlip(prob=0.5, direction="horizontal"),
        transforms.RandomChoiceRotate(angles=[0, 90, 180, 270]),
        transforms.PhotoMetricDistortion(),
    ]

    image_path = "./data/poquets/images/test/DJI_20240529102221_0010_V.jpg"

    image = Image.open(image_path).convert("RGB")

    # Convert to numpy array
    image_np = np.array(image)

    # Show the image using the utility function
    start = time.time()
    for i in range(25):
        image_dict = {"img": image_np}
        for transform in pipeline_new:
            # Apply each transformation
            image_dict = transform(image_dict)
            
        # transformed_image = pipeline[1]({"img": image_np})
        # save transformed image
        transformed_image_pil = Image.fromarray(image_dict["img"])
        transformed_image_pil.save(f"./temp/swin/transformed_{i}.jpg")

    end = time.time()
    print(f"Time taken for 25 transformations: {end - start} seconds")
    