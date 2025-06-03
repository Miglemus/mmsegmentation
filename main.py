if __name__ == "__main__":
    import cv2
    import numpy as np
    from glob import glob

    mask_paths = glob('data/poquets_complet/annotations/train/*.png')

    for path in mask_paths[:5]:  # check 5 masks
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(f'{path} - Unique labels:', np.unique(mask))

