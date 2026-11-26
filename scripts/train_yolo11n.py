import random
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

RAW_IMG_DIR = Path("dataset/raw_images")
RAW_LBL_DIR = Path("dataset/raw_labels")
OUT_DIR = Path("dataset/yolo")
MODEL_OUT = Path("models/yolo11n_ryl.onnx")

CLASSES = ["gold_rustler", "gold_bag", "player", "dead_rustler"]


def prepare_dataset(val_ratio: float = 0.2, seed: int = 42) -> Path:
    images = sorted(
        p for p in RAW_IMG_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"Tidak ada gambar di {RAW_IMG_DIR}")

    random.seed(seed)
    random.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio))
    splits = {"val": images[:val_count], "train": images[val_count:]}

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split, items in splits.items():
        img_dst = OUT_DIR / "images" / split
        lbl_dst = OUT_DIR / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in items:
            shutil.copy2(img_path, img_dst / img_path.name)
            src_lbl = RAW_LBL_DIR / f"{img_path.stem}.txt"
            dst_lbl = lbl_dst / f"{img_path.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

    yaml_path = OUT_DIR / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    yaml_path.write_text(
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"Dataset siap: {len(splits['train'])} train, {len(splits['val'])} val")
    return yaml_path


def main() -> None:
    yaml_path = prepare_dataset()
    train_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training menggunakan device: {train_device}")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=16,
        device=train_device,
        patience=25,
        mosaic=0.5,
        fliplr=0.5,
        flipud=0.0,
        project="runs/ryl_yolo11n",
        name="exp",
        exist_ok=True,
    )

    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))
    exported = best_model.export(format="onnx", imgsz=640, simplify=True, opset=12)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, MODEL_OUT)
    print(f"Model ONNX siap dipakai VM: {MODEL_OUT}")


if __name__ == "__main__":
    main()
