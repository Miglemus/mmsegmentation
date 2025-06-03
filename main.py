if __name__ == "__main__":

    from utils.train import train


    config_file = "./configs/my_configs/mask2former_r50_8xb2_poquets-complet-512x512.py"

    train(config_file)
