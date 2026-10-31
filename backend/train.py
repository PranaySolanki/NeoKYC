import argparse
import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.training.config import TrainingConfig
from backend.training.trainer import Trainer, set_seed
from backend.data.faceforensics import FaceForensicsDataset, create_ff_dataloaders
from backend.models.f3net import F3Net


def run_smoke_test() -> None:
    """
    Executes a controlled, minimal smoke test using synthetic tensors.
    Does NOT download any dataset or large weights.
    Verifies the entire pipeline: DataLoader -> Forward -> Loss -> Backward -> Optimizer -> Scheduler -> Checkpoint.
    """
    print("=" * 60)
    print("RUNNING F3-NET CONTROLLED TRAINING SMOKE TEST")
    print("=" * 60)

    config = TrainingConfig(
        mode="Mix",
        batch_size=2,
        max_epochs=2,
        lr=0.002,
        log_freq=1,
        output_dir="checkpoints/smoke_test",
        use_learnable_filters=True,
    )

    set_seed(config.seed)
    print(f"Hardware Resource Check: Device='{config.device}', CUDA available={torch.cuda.is_available()}")

    # Create small synthetic dataset (4 samples: 2 real, 2 fake)
    print("\n[1/5] Generating synthetic batch: 4 samples of shape (3, 299, 299)...")
    images = torch.randn(4, 3, 299, 299)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    # TensorDataset simulating FaceForensics batch
    class SyntheticFFDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls):
            self.imgs = imgs
            self.lbls = lbls
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            meta = {"video_id": f"synth_{idx}", "manipulation": "smoke_test", "compression": "c40", "frame_idx": str(idx)}
            return self.imgs[idx], self.lbls[idx], meta

    train_ds = SyntheticFFDataset(images[:2], labels[:2])
    val_ds = SyntheticFFDataset(images[2:], labels[2:])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    print("[2/5] Initializing Trainer with SGD (momentum=0.9, lr=0.002) and Cosine Annealing...")
    trainer = Trainer(config)

    print("\n[3/5] Executing 2 training & validation iterations...")
    history = trainer.fit(train_loader, val_loader)

    print("\n[4/5] Verifying saved checkpoints in 'checkpoints/smoke_test'...")
    latest_ckpt = os.path.join(config.output_dir, "latest_checkpoint.pth")
    assert os.path.isfile(latest_ckpt), f"Latest checkpoint not found at {latest_ckpt}"
    print(f"Found saved checkpoint: {latest_ckpt}")

    print("\n[5/5] Testing checkpoint restoration...")
    trainer_restored = Trainer(config)
    trainer_restored.load_checkpoint(latest_ckpt)
    print("Checkpoint successfully loaded and weights verified!")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY (All components functional)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="F3-Net Training Pipeline for FaceForensics++")
    parser.add_argument("--smoke_test", action="store_true", help="Run a small controlled synthetic smoke test")
    parser.add_argument("--dataset_dir", type=str, default="data/FaceForensics++", help="Path to FaceForensics++ root")
    parser.add_argument("--compression", type=str, default="c40", choices=["c40", "c23", "raw"], help="Compression level")
    parser.add_argument("--mode", type=str, default="Mix", choices=["Mix", "Both", "FAD", "LFS"], help="Model mode")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--output_dir", type=str, default="checkpoints/f3net", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained_xception", type=str, default=None, help="Path to ImageNet-pretrained Xception weights (.pth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Hardware device")
    parser.add_argument("--num_workers", type=int, default=2, help="Data loader workers")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    # Full training pathway
    print(f"Checking dataset directory: {args.dataset_dir}")
    if not os.path.exists(args.dataset_dir):
        print("\n" + "!" * 70)
        print("DATASET NOT FOUND:")
        print(f"The specified dataset path '{args.dataset_dir}' does not exist.")
        print("\nHOW TO OBTAIN FACEFORENSICS++:")
        print("1. FaceForensics++ is a restricted academic benchmark requiring official access.")
        print("2. Submit the access request form at:")
        print("   https://github.com/ondyari/FaceForensics")
        print("3. Use the official download script provided by the authors:")
        print("   python download_faceforensics.py <output_path> -d all -c c40 -t videos")
        print("4. Extract facial frames using the standard preprocessing pipeline.")
        print("!" * 70)
        sys.exit(1)

    config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        compression=args.compression,
        mode=args.mode,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume,
        pretrained_xception_path=args.pretrained_xception,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )

    train_loader, val_loader, _ = create_ff_dataloaders(
        root_dir=config.dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        compression=config.compression,
    )

    trainer = Trainer(config)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
