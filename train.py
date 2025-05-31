import orbax.checkpoint
import cv2
from flax import nnx
import optax
import orbax.checkpoint as ocp
from jax.tree_util import tree_map
import jax.numpy as jnp
import albumentations as A
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data.dataloader import default_collate, DataLoader
import random
import numpy as np
from tqdm import tqdm
from models.efficientnet import efficientnet_v2_s

params = {
    "num_epochs": 5,
    "batch_size": 32,
    "target_size": 224,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data sample into a batch.
    default collate creates pytorch tensors, then tree_map converts them to numpy arrays.
    """
    return tree_map(jnp.asarray, default_collate(batch))


def open_img(img_path):
    # img = Image.open(img_path)
    # img = img.convert("RGB")  # Ensure the image is in RGB format
    # return np.asarray(img)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    return img


class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return self.transforms(image=image)["image"]


train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=params["target_size"], p=1.0),
        A.RandomCrop(height=params["target_size"], width=params["target_size"], p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
    ]
)

val_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=params["target_size"], p=1.0),
        A.CenterCrop(height=params["target_size"], width=params["target_size"], p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
    ]
)

train_ds = torchvision.datasets.ImageFolder(
    "./MpoxData/Train",
    transform=Transforms(train_transforms),
    loader=open_img,
)
val_ds = torchvision.datasets.ImageFolder(
    "./MpoxData/Valid", transform=Transforms(val_transforms), loader=open_img
)


train_loader = DataLoader(
    train_ds,
    batch_size=params["batch_size"],
    shuffle=True,
    collate_fn=numpy_collate,
    num_workers=0,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=params["batch_size"],
    shuffle=False,
    collate_fn=numpy_collate,
    num_workers=0,
    pin_memory=True,
)


def create_model(seed: int):
    return efficientnet_v2_s(rngs=nnx.Rngs(seed), num_classes=6)


def create_and_save(seed: int, path: str):
    model = create_model(seed)
    state = nnx.state(model)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(f"{path}/state", state)


def load_model(path: str):
    model = nnx.eval_shape(lambda: create_model(0))
    state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(f"{path}/state", item=state)

    nnx.update(model, state)
    return model


# 修改模型创建和加载的部分
model = create_model(0)

optimizer = nnx.Optimizer(
    model,
    optax.adamw(
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
    ),
)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch[1],
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(grads)


@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

best_acc = -1

for epoch in range(params["num_epochs"]):
    # Training
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Traning"):
        train_step(model, optimizer, metrics, batch)
    train_result = metrics.compute()
    for k, v in train_result.items():
        metrics_history[f"train_{k}"].append(v)
    metrics.reset()
    print(
        f"✅ Train Loss: {train_result['loss']:.4f}, Acc: {train_result['accuracy'] * 100:.6f}%"
    )

    # Validation
    model.eval()
    for val_batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
        eval_step(model, metrics, val_batch)
    val_result = metrics.compute()
    for k, v in val_result.items():
        metrics_history[f"val_{k}"].append(v)
    metrics.reset()
    print(
        f"📊 Val Loss: {val_result['loss']:.4f}, Acc: {val_result['accuracy'] * 100:.6f}%"
    )
    if val_result["accuracy"] > best_acc:
        best_acc = val_result["accuracy"]
        checkpoint_path = f"/Users/billy/Documents/DLStudy/JaxVision/checkpoints/best_model_Epoch_{epoch + 1}_Acc_{best_acc:.6f}/state"
        checkpointer = ocp.PyTreeCheckpointer()
        state = nnx.state(model)
        checkpointer.save(checkpoint_path, state)
        print(f"🎉 New best model saved with accuracy: {best_acc * 100:.6f}%")
# Plotting after training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title("Loss")
ax2.set_title("Accuracy")

for dataset in ("train", "val"):
    ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
    ax2.plot(metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy")

ax1.set_xlabel("Evaluation Step")
ax2.set_xlabel("Evaluation Step")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
ax1.legend()
ax2.legend()
plt.show()
