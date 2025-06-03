import wandb
import mmengine
from mmengine.runner import Runner
from mmengine.config import Config
import os

def train_sweep(cfg_path):
    cfg = Config.fromfile(cfg_path)

    # Override config from WandB sweep

    # change the learning rate
    cfg.optimizer.lr = wandb.config.lr

    # change the batchzise
    cfg.data.train_dataloader.batch_size = wandb.config.batch_size

    # change the backbone architecture
    cfg.model.backbone.depth = wandb.config.depth

    if wandb.config.depth == 101:
        cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet101')
    else:
        cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet50')

    # iterations
    cfg.train_cfg.max_iters = wandb.config.max_iters

    # change the class weights
    class_weight = [0.0, 0.0, 0.1]

    # bg class weight
    class_weight[0] = wandb.config.bg_weight

    # poquet class weight
    class_weight[1] = wandb.config.poquet_weight

    cfg.model.decode_head.loss_cls.class_weight = class_weight


    # You can also dynamically change model type, backbone, etc., with caution
    cfg.work_dir = f"work_dirs/sweep_{wandb.run.name}"
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Set logger hook to include wandb
    cfg.default_hooks.logger.wandb = dict(
        type='WandbLoggerHook',
        init_kwargs=dict(
            project='sweep-example',
            name=wandb.run.name,
        )
    )

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == "__main__":
    wandb.init(project="mmsegmentation")
    train_sweep("../configs/my_configs/mask2former_r50_8xb2_poquets-complet-512x512.py")
