if __name__ == "__main__":

    # from utils.train import train


    # config_file = "./configs/my_configs/mask2former_r50_8xb2_poquets-complet-512x512.py"

    # train(config_file)

    import os
    from utils.visualization import show_image
    from mmseg.datasets.transforms import transforms
    from mmcv.transforms import RandomChoiceResize, CenterCrop
    from random import sample
    from PIL import Image
    import numpy as np
    import time

    pipeline_new = [
        transforms.RandomMosaic(prob=1.0, img_scale=(1978/2, 2640/2)),
        transforms.RandomUniformDownScale(min_down_scale=3.0, max_down_scale=5.0),
        transforms.RandomCrop(crop_size=(640, 640)),
        transforms.RandomFlip(prob=0.5, direction="vertical"),
        transforms.RandomFlip(prob=0.5, direction="horizontal"),
        # transforms.RandomRotate(degree=(0, 360), prob=1.0,),
        transforms.PhotoMetricDistortion(),
    ]

    image_path = "./data/poquets/images/test/DJI_20240529102221_0010_V.jpg"

    # Your main image
    main_image = Image.open(image_path).convert("RGB")
    main_image_np = np.array(main_image)

    # Load extra images for mosaic
    image_folder = "./data/poquets/images/test"
    image_list = sorted([
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.endswith(".jpg") and f != os.path.basename(image_path)
    ])
    # Ensure you have enough images to sample from
    assert len(image_list) >= 3, "Need at least 3 more images for mosaic!"

    start = time.time()
    for i in range(25):
        # Sample 3 other images for mix_results
        mix_images = sample(image_list, 3)
        mix_results = []
        for path in mix_images:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
            mix_results.append({"img": img_np})

        # Prepare the input dict for transforms
        image_dict = {
            "img": main_image_np,
            "mix_results": mix_results,
            "seg_fields": [],  # if you have labels, add them here
        }

        for transform in pipeline_new:
            image_dict = transform(image_dict)

        transformed_image_pil = Image.fromarray(image_dict["img"])
        transformed_image_pil.save(f"./temp/swin_old/transformed_{i}.jpg")

    end = time.time()
    print(f"Time taken for 25 transformations: {end - start:.2f} seconds")

    