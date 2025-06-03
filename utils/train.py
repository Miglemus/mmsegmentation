import mmcv
import mmengine
from mmengine.runner import Runner

def train(cfg_path):
    cfg = mmengine.Config.fromfile(cfg_path)

    cfg.work_dir="work_dirs"
    
    print(f"Config:\n{cfg.pretty_text}")

    runner = Runner.from_cfg(cfg)

    runner.train()
