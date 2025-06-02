import re
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import jax.numpy as jnp
import orbax.checkpoint as ocp
import albumentations as A
from flax import nnx
from jax.tree_util import tree_map

from models.resnet import resnet18


class Config:
    """集中管理常量参数"""
    TARGET_SIZE = 224
    NUM_CLASSES = 6
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    CHECKPOINT_DIR = Path("/Users/billy/Documents/DLStudy/JaxVision/checkpoints")
    CLASS_NAMES = [
        "Chickenpox",
        "Cowpox",
        "Healthy",
        "HFMD",
        "Measles",
        "Monkeypox",
    ]
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def numpy_collate(batch):
    """
    Custom collate: PyTorch tensors -> numpy arrays (for JAX)
    """
    # default_collate 会把所有张量变成一个 batch 的 tensor
    # 然后我们用 tree_map 把 Tensor -> numpy
    return tree_map(lambda x: x.numpy() if isinstance(x, torch.Tensor) else x,
                    default_collate(batch))


def load_image_rgb(img_path: Path) -> np.ndarray:
    """
    Load image in RGB format. 如果读不到，抛出异常。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_test_transform(target_size: int) -> A.Compose:
    """
    返回仅包含 Resize 和 Normalize 的测试/推理时用的 Albumentations pipeline
    """
    return A.Compose([
        A.Resize(height=target_size, width=target_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
    ])


class PredictionDataset(torch.utils.data.Dataset):
    """
    数据集：遍历一个目录下所有图片，加载、应用 transform 并返回 (numpy_image, path, name)
    """

    def __init__(self, image_dir: Path, transforms: A.Compose = None):
        self.image_dir = image_dir
        self.transforms = transforms

        # 用 rglob 一次性匹配所有后缀 (大小写)
        self.image_paths = []
        for ext in Config.VALID_EXTENSIONS:
            pattern = f"*{ext}"
            self.image_paths += list(image_dir.rglob(pattern))
            self.image_paths += list(image_dir.rglob(pattern.upper()))

        # 保证顺序一致
        self.image_paths = sorted(set(self.image_paths))
        if not self.image_paths:
            raise ValueError(f"目录下没有找到有效图片: {image_dir}")

        print(f"🗂️ 共找到 {len(self.image_paths)} 张图片: {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = load_image_rgb(path)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        # 返回 (numpy_image, str(path), image_name)
        return img, str(path), path.name


def create_dataloader(image_dir: str, batch_size: int, num_workers: int, target_size: int):
    """
    直接创建 Dataset + DataLoader
    """
    tfm = build_test_transform(target_size)
    dataset = PredictionDataset(Path(image_dir), transforms=tfm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def create_model(seed: int, num_classes: int):
    """
    JAX + Flax 下创建 ResNet18 模型实例（参数 shape 用于初始化）
    """
    return resnet18(rngs=nnx.Rngs(seed), num_classes=num_classes)


def load_model_from_checkpoint(ckpt_path: str, num_classes: int):
    """
    从 checkpoint 恢复模型参数，并切换到 eval 模式
    """
    # 先用 eval_shape 构造一个同结构的“空” model
    model = nnx.eval_shape(lambda: create_model(0, num_classes))
    state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(ckpt_path, item=state)
    nnx.update(model, state)
    model.eval()
    return model




def find_best_checkpoint(checkpoint_dir: Path) -> str:
    """
    在checkpoint目录里找带 'best_model' 且 acc 最大的文件夹，
    返回该文件夹下的 'state' 路径
    """
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint 目录不存在: {checkpoint_dir}")

    # 所有名字包含 best_model 的子目录
    candidates = [d for d in checkpoint_dir.iterdir() if d.is_dir() and "best_model" in d.name]
    if not candidates:
        raise ValueError(f"没找到 any 'best_model' 子目录: {checkpoint_dir}")

    # 用正则提取 “Acc_0.952000” 里面的小数
    def extract_acc(name: str) -> float:
        m = re.search(r"Acc_([0-9]*\.?[0-9]+)", name)
        return float(m.group(1)) if m else 0.0

    # 按 acc 降序排列
    candidates.sort(key=lambda d: extract_acc(d.name), reverse=True)
    best_dir = candidates[0]
    state_file = best_dir / "state"
    print(f"🎯 选择 checkpoint: {best_dir.name}")
    return str(state_file)


def preprocess_one(image_path: str, transform: A.Compose):
    """
    单张图片预处理: 读图 -> 变换 -> 扩展 batch 维度 -> 转 JAX array
    """
    img = load_image_rgb(Path(image_path))
    img_t = transform(image=img)["image"]
    # 转为 float32 并在 axis=0 添加 batch 维度
    return jnp.expand_dims(jnp.array(img_t, dtype=jnp.float32), axis=0), img


@nnx.jit
def predict_batch(model, images: jnp.ndarray):
    """
    JIT 编译的预测接口: 
    输入: (batch, H, W, C)
    返回: (pred_idx, 全概率分布)
    """
    logits = model(images)
    probs = nnx.softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1)
    return preds, probs


def predict_single(model, image_path: str, transform: A.Compose, class_names: list):
    """
    单张图片推理，返回 dict 包含预测结果、原图
    """
    img_batch, orig_img = preprocess_one(image_path, transform)
    pred_idx, probs = predict_batch(model, img_batch)

    idx = int(pred_idx[0])
    return {
        "pred_idx": idx,
        "pred_name": class_names[idx],
        "confidence": float(probs[0, idx]),
        "all_probs": np.array(probs[0]),
        "orig_img": orig_img,
        "image_path": image_path,
        "image_name": Path(image_path).name,
    }


def predict_directory(model, image_dir: str, class_names: list,
                      batch_size: int, num_workers: int, save_csv: bool = False):
    """
    对目录下所有图片进行推理，使用 DataLoader 批量处理
    """
    print(f"\n🗂️ 对目录进行推理: {image_dir}")
    loader = create_dataloader(image_dir, batch_size, num_workers, Config.TARGET_SIZE)

    all_results = []
    for images_np, paths, names in tqdm(loader, desc="批量推理中"):
        # 样本已经是 numpy-array，shape=(batch, H, W, C)
        # 直接转 JAX array
        images_jax = jnp.array(images_np, dtype=jnp.float32)
        preds, probs = predict_batch(model, images_jax)

        # 遍历 batch 内的每张图，收集结果
        for i in range(len(paths)):
            idx = int(preds[i])
            all_results.append({
                "image_path": paths[i],
                "image_name": names[i],
                "pred_idx": idx,
                "pred_name": class_names[idx],
                "confidence": float(probs[i, idx]),
                "all_probs": np.array(probs[i]),
            })

    _print_summary(all_results)
    if save_csv:
        _save_results_csv(all_results, image_dir, class_names)
    return all_results


def _print_summary(results: list):
    """
    打印预测总结信息：总数、平均置信度、各类分布 & 最高/最低置信样本
    """
    total = len(results)
    if total == 0:
        print("⚠️ 没有结果可展示")
        return

    avg_conf = sum(r["confidence"] for r in results) / total
    print("\n📊 预测总结：")
    print(f"  总图像数量: {total}")
    print(f"  平均置信度: {avg_conf:.4f}")

    # 各类计数
    class_counts = {}
    for r in results:
        class_counts[r["pred_name"]] = class_counts.get(r["pred_name"], 0) + 1

    print("  类别分布：")
    for cls, cnt in sorted(class_counts.items()):
        pct = cnt / total * 100
        print(f"    {cls}: {cnt} ({pct:.1f}%)")

    # 排序找最高 / 最低
    sorted_r = sorted(results, key=lambda x: x["confidence"], reverse=True)
    print("\n🎯 最高置信度样本：")
    for i, r in enumerate(sorted_r[:3], 1):
        print(f"    {i}. {r['image_name']}: {r['pred_name']} ({r['confidence']:.4f})")

    print("\n🤔 最低置信度样本：")
    for i, r in enumerate(sorted_r[-3:], 1):
        print(f"    {i}. {r['image_name']}: {r['pred_name']} ({r['confidence']:.4f})")




def _save_results_csv(results: list, image_dir: str, class_names: list):
    """
    将预测结果保存到 CSV 文件，包含每个类别的概率列
    """
    out_path = Path(image_dir) / "prediction_results.csv"
    print(f"\n💾 保存结果到: {out_path}")

    fieldnames = ["image_name", "image_path", "pred_name", "confidence"]
    # 添加每个 class 的概率列
    for c in class_names:
        fieldnames.append(f"prob_{c}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "image_name": r["image_name"],
                "image_path": r["image_path"],
                "pred_name": r["pred_name"],
                "confidence": f"{r['confidence']:.6f}",
            }
            for idx, cn in enumerate(class_names):
                row[f"prob_{cn}"] = f"{r['all_probs'][idx]:.6f}"
            writer.writerow(row)

    print("✅ 保存完成！")


def visualize_results_sample(results: list, num_samples: int = 9):
    """
    从最高与最低置信度各取部分样本，显示图 + 预测结果
    """
    if not results:
        print("⚠️ 无结果可视化")
        return

    sorted_r = sorted(results, key=lambda x: x["confidence"], reverse=True)
    top_k = sorted_r[: num_samples // 2]
    low_k = sorted_r[-(num_samples - num_samples // 2):]
    samples = top_k + low_k

    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # 确保 axes 是二维数组形式
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, r in enumerate(samples):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        try:
            img = load_image_rgb(Path(r["image_path"]))
            ax.imshow(img)
            title = f"{r['image_name']}\n{r['pred_name']}\nConf: {r['confidence']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        except Exception:
            ax.text(0.5, 0.5, f"加载失败\n{r['image_name']}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{r['pred_name']}\nConf: {r['confidence']:.3f}")
            ax.axis("off")

    # 隐藏多余子图
    total_plots = rows * cols
    for i in range(len(samples), total_plots):
        row, col = divmod(i, cols)
        axes[row, col].axis("off")

    plt.suptitle("示例预测结果（高/低置信度）", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_prediction(result: dict, show_prob: bool = True):
    """
    单张图片可视化：图像 + 条形图概率分布
    """
    figsize = (15, 6) if show_prob else (8, 6)
    fig, axs = plt.subplots(1, 2 if show_prob else 1, figsize=figsize)
    if not show_prob:
        axs = [axs]

    # 左图：原图 + 标题
    img = result.get("orig_img", load_image_rgb(Path(result["image_path"])))
    axs[0].imshow(img)
    axs[0].set_title(f"{result['image_name']}\nPred: {result['pred_name']}\nConf: {result['confidence']:.4f}")
    axs[0].axis("off")

    if show_prob:
        # 右图：条形图表示各类别概率
        classes = Config.CLASS_NAMES
        probs = result["all_probs"]
        bars = axs[1].bar(range(len(classes)), probs)
        axs[1].set_xticks(range(len(classes)))
        axs[1].set_xticklabels(classes, rotation=45, ha="right")
        axs[1].set_xlabel("类别")
        axs[1].set_ylabel("概率")
        axs[1].set_title("各类概率分布")
        # 高亮预测类别
        bars[result["pred_idx"]].set_color("red")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="使用训练好的 ResNet18 模型进行推理")
    parser.add_argument("--image", type=str, help="单张图片路径，用于预测")
    parser.add_argument("--images", type=str, nargs="+", help="多张图片路径列表")
    parser.add_argument("--directory", type=str, help="图片目录，用于批量预测")
    parser.add_argument("--checkpoint", type=str, help="指定 checkpoint 文件 (可选)")
    parser.add_argument("--checkpoint_dir", type=str, default=str(Config.CHECKPOINT_DIR),
                        help="checkpoint 所在目录")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="批量预测时的 batch size")
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS, help="DataLoader 的 num_workers")
    parser.add_argument("--save_results", action="store_true", help="是否保存预测结果到 CSV")
    parser.add_argument("--no_viz", action="store_true", help="跳过可视化")
    parser.add_argument("--class_names", type=str, nargs="+", default=Config.CLASS_NAMES, help="类别名称列表")

    args = parser.parse_args()

    # 至少提供一个输入选项
    if not (args.image or args.images or args.directory):
        print("❌ 请选择 --image / --images / --directory 中的一种参数来执行预测")
        return

    print("🔮 开始预测流程...")

    # 选择 checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        try:
            ckpt_path = find_best_checkpoint(Path(args.checkpoint_dir))
        except Exception as e:
            print(f"❌ 获取最佳 checkpoint 失败: {e}")
            return

    # 加载模型
    print("\n🏗️ 正在加载模型...")
    try:
        model = load_model_from_checkpoint(ckpt_path, Config.NUM_CLASSES)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载出错: {e}")
        return

    # 统一 transform
    transform = build_test_transform(Config.TARGET_SIZE)

    # 单张图片
    if args.image:
        print(f"\n🔍 正在对单张图片进行预测: {args.image}")
        try:
            res = predict_single(model, args.image, transform, args.class_names)
            print("\n📊 预测结果：")
            print(f"  图像: {res['image_path']}")
            print(f"  预测类别: {res['pred_name']}")
            print(f"  置信度: {res['confidence']:.4f}")
            print(f"  类别索引: {res['pred_idx']}")

            if not args.no_viz:
                visualize_prediction(res)
        except Exception as e:
            print(f"❌ 单图预测失败: {e}")

    # 多张图片逐一预测
    elif args.images:
        print(f"\n🔍 正在对多张图片列表进行预测，共 {len(args.images)} 张")
        results = []
        for img_path in args.images:
            try:
                r = predict_single(model, img_path, transform, args.class_names)
                results.append(r)
                print(f"✅ {r['image_name']}: {r['pred_name']} ({r['confidence']:.4f})")
            except Exception as e:
                print(f"❌ {Path(img_path).name} 预测失败: {e}")

        if results:
            print("\n📊 批量预测统计：")
            dist = {}
            for r in results:
                dist[r["pred_name"]] = dist.get(r["pred_name"], 0) + 1
            for cls, cnt in dist.items():
                print(f"  {cls}: {cnt}")

            if not args.no_viz and results:
                print("🔎 显示第一张预测结果可视化")
                visualize_prediction(results[0])

    # 整个目录
    elif args.directory:
        print(f"\n🔍 正在对目录进行批量预测: {args.directory}")
        try:
            all_results = predict_directory(
                model,
                args.directory,
                args.class_names,
                args.batch_size,
                args.num_workers,
                save_csv=args.save_results
            )
            if not args.no_viz and all_results:
                print("\n📈 展示示例可视化")
                visualize_results_sample(all_results)
        except Exception as e:
            print(f"❌ 目录预测出错: {e}")
            return


if __name__ == "__main__":
    main()
