if __name__ == "__main__":

    # from utils.train import train


    # config_file = "./configs/my_configs/mask2former_r50_8xb2_poquets-complet-512x512.py"

    # train(config_file)

    import os
    from utils.visualization import show_image
    from mmseg.datasets.transforms import transforms
    from PIL import Image
    import numpy as np

    pipeline = [
        transforms.RandomCrop(crop_size=(906, 906)),
        transforms.RandomRotate(degree=(0, 360), prob=1.0,),
        transforms.RandomCrop(crop_size=(640, 640)),
    ]

    image_path = "./data/poquets/images/test/DJI_20240529102221_0010_V.jpg"

    image = Image.open(image_path).convert("RGB")

    # Convert to numpy array
    image_np = np.array(image)

    # Show the image using the utility function
    for i in range(10):
        iamge_dict = {"img": image_np}
        for transform in pipeline:
            # Apply each transformation
            iamge_dict = transform(iamge_dict)
            
        # transformed_image = pipeline[1]({"img": image_np})
        # save transformed image
        transformed_image_pil = Image.fromarray(iamge_dict["img"])
        transformed_image_pil.save(f"./temp/test/transformed_{i}.jpg")

    