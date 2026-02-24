import argparse
import os
import subprocess
from pathlib import Path

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner

from configs import entrypoint_inference

TOOLS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
REPO_DIR = TOOLS_DIR.parent
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMDet3D test (and eval) a model",
    )
    parser.add_argument(
        "--checkpoint",
        help="checkpoint file",
        default="work_dirs/clean_forestformer/epoch_3000_fix.pth",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="the directory to save the file containing evaluation metrics",
        default=Path("work_dirs/entrypoint_inference"),
    )
    parser.add_argument(
        "--data",
        "-d",
        nargs="+",
        help="Input data for segmentation",
        type=Path,
    )
    parser.add_argument(
        "--skip-data-preparation",
        action="store_true",
        help="Skip data preparation",
    )
    parser.add_argument("--max-points", type=int, default=640_000)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    return args


def prepare_data(data: list[Path] | None):
    if data is None:
        data = []
    data_names = [d.stem for d in data]
    print("-" * 100)
    print("Preparing data")
    print("-" * 100)
    meta_data_dir = REPO_DIR / "data" / "ForAINetV2" / "meta_data"
    with open(meta_data_dir / "test_list.txt", "w") as f:
        f.writelines(data_names)
    with open(meta_data_dir / "train_list.txt", "w") as f:
        pass
    with open(meta_data_dir / "val_list.txt", "w") as f:
        pass
    data_dir = REPO_DIR / "data" / "ForAINetV2" / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in data_dir.iterdir():
        print(f"Removing old test file: {path}")
        os.remove(str(path))
    for d in data:
        os.symlink(
            str(d.resolve()),
            str(data_dir / d.name),
        )

    print("-" * 100)
    print("Running batch loading")
    print("-" * 100)
    batch_process = subprocess.Popen(
        "python batch_load_ForAINetV2_data.py".split(),
        cwd=REPO_DIR / "data" / "ForAINetV2",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    for line in batch_process.stdout:
        print(line)
    batch_process.wait()

    print("-" * 100)
    print("Runinng data creation")
    print("-" * 100)
    create_data_process = subprocess.Popen(
        "python tools/create_data_forainetv2.py forainetv2".split(),
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ,
    )
    for line in create_data_process.stdout:
        print(line)
    create_data_process.wait()


def segmentation(
    work_dir: Path,
    checkpoint: str,
    radius: int,
    max_points: int,
    chunk_size: int,
):
    work_dir.mkdir(exist_ok=True, parents=True)
    cfg = Config.fromfile(REPO_DIR / "configs" / "entrypoint_inference.py")
    cfg.load_from = checkpoint
    cfg.work_dir = str(work_dir.resolve())
    cfg.model.test_cfg["output_dir"] = cfg.work_dir
    cfg.model.chunk = chunk_size
    cfg.model.num_points = max_points
    cfg.model.radius = radius
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_data_preparation:
        prepare_data(args.data)
    output_directory = segmentation(
        args.work_dir,
        args.checkpoint,
        args.radius,
        args.max_points,
        args.chunk_size,
    )
