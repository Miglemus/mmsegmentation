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

    pipeline_new = [ # downscale 3-5 times, randomcrop 640x640, randomflip horizontal and vertical, RandomOrthogonalRotate, photometric distortion
        transforms.RandomUniformDownScale(min_down_scale=3.0, max_down_scale=5.0),
        transforms.RandomCrop(crop_size=(640, 640)),
        transforms.RandomFlip(prob=0.5, direction="vertical"),
        transforms.RandomFlip(prob=0.5, direction="horizontal"),
        transforms.RandomRotate(degree=(0, 360), prob=1.0,),
        transforms.PhotoMetricDistortion(),
    ]

    pipeline_swin = [
        transforms.RandomCrop(crop_size=(906, 906)),
        transforms.RandomFlip(prob=0.5, direction="vertical"),
        transforms.RandomFlip(prob=0.5, direction="horizontal"),
        transforms.RandomRotate(degree=(0, 360), prob=1.0,),
        CenterCrop(crop_size=(640, 640)),
        transforms.PhotoMetricDistortion(
            brightness_delta=16,
            contrast_range=(0.75, 1.3),
            hue_delta=9,
            saturation_range=(0.5, 1.5)
        )
    ]

    pipeline_r50 = [
        RandomChoiceResize(
            max_size=2048,
            resize_type=transforms.ResizeShortestEdge,
            scales=[
                512,
                614,
                716,
                819,
                921,
                1024,
                1126,
                1228,
                1331,
                1433,
                1536,
                1638,
                1740,
                1843,
                1945,
                2048,
            ],
       ),
        transforms.RandomCrop(crop_size=(512, 512)),
        transforms.RandomFlip(prob=0.5),
        transforms.PhotoMetricDistortion()
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
        transformed_image_pil.save(f"./temp/swin_old/transformed_{i}.jpg")

    end = time.time()
    print(f"Time taken for 25 transformations: {end - start} seconds")
    