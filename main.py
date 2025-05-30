if __name__ == "__main__":
    import os
    from pycocotools.coco import COCO
    import utils.utils

    missing_files = [
        "DJI_20240529132142_0044_V.jpg",
        "DJI_20240529131710_0025_V.jpg",
        "DJI_20240529125337_0050_V.jpg",
        "DJI_20240529124844_0061_V.jpg",
        "DJI_20240529132127_0027_V.jpg"
    ]


    annotation_folder = "./data/poquets/annotations"
    images_folder = "./data/poquets/images"


    for image_set in ["test", "train", "val"]:
        coco_annot_path = os.path.join(annotation_folder, f"{image_set}_json/{image_set}_annot.json")
        coco = COCO(coco_annot_path)

        for missing_image in missing_files:            
            for coco_annot_image in coco.imgs.values():
                if missing_files == coco_annot_image["file_name"]:
                    print(image_set)
            