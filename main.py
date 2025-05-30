if __name__ == "__main__":
    import os


    annotation_folder = "./data/cityscapes/gtFine"
    image_folder = "./data/cityscapes/leftImg8bit"
    no_annotation_folder = "./data/cityscapes/no_annotation"
    no_image_folder = "./data/cityscapes/no_image"

    absolute_annotations = set()
    absolute_images = set()

    # iterate over all the annotations
    for image_set in os.listdir(annotation_folder):
        for city_folder in os.listdir(os.path.join(annotation_folder, image_set)):
            for file_name in os.listdir(os.path.join(annotation_folder, image_set, city_folder)):
                file_path = os.path.join(annotation_folder, image_set, city_folder, file_name)
                without_ext = os.path.splitext(file_path)[0]
                basename = os.path.basename(without_ext)
                absolute_instance = basename.split("_")[:3]

                absolute_annotations.add("_".join(absolute_instance))

    # iterate over all the images
    for image_set in os.listdir(image_folder):
        for city_folder in os.listdir(os.path.join(image_folder, image_set)):
            for file_name in os.listdir(os.path.join(image_folder, image_set, city_folder)):
                file_path = os.path.join(image_folder, image_set, city_folder, file_name)
                without_ext = os.path.splitext(file_path)[0]
                basename = os.path.basename(without_ext)
                absolute_instance = basename.split("_")[:3]

                absolute_images.add("_".join(absolute_instance))

    print(f"length of absolute annotations: {len(absolute_annotations)}")
    print(f"length of absolute images: {len(absolute_images)}")

    assert (len(absolute_annotations) == len(absolute_images)), "Mismatch in number of annotations and images"

    for instance in absolute_annotations:
        if instance not in absolute_images:
            assert instance in absolute_images, f"Annotation {instance} does not have a corresponding image"

    print("all annotations have corresponding images")
    print(absolute_annotations)
