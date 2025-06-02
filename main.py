if __name__ == "__main__":
    from mmseg.apis import MMSegInferencer, init_model, inference_model

    config_path = "configs/my_configs/mask2former_r50_8xb2-90k_cityscapes-512x1024.py"
    checkpoint_path = "work_dirs/mask2former_r50_8xb2-90k_cityscapes-512x1024/best_mioU_iter_470"

    img_dir = "data/poquets/images/no_corresponding_annot"


