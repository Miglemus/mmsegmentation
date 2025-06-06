if __name__ == "__main__":

    # from utils.train import train


    # config_file = "./configs/my_configs/mask2former_r50_8xb2_poquets-complet-512x512.py"

    # train(config_file)

    import os
    from utils.visualization import show_image

    image_dir = "./data/poquets_complet/images/train"
    mask_dir = "./data/poquets_complet/annotations/train"

    images = [os.path.join(image_dir, image) for image in sorted(os.listdir(image_dir))]
    masks = [os.path.join(mask_dir, mask) for mask in sorted(os.listdir(mask_dir))]

    show_image(images, masks)