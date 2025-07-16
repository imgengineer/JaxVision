
JAXvision is a new library that brings a collection of popular computer vision models, inspired by the familiar torchvision.models API, to the JAX ecosystem using Flax NNX. If you're accustomed to the ease of use and readily available architectures in PyTorch's torchvision, JAXvision offers a similar experience for JAX practitioners, enabling you to build and experiment with state-of-the-art vision models directly within the functional and high-performance environment of JAX.


- Seamless JAX Integration: Built natively with Flax NNX, Jaxvision models leverage JAX's powerful features like JIT compilation, automatic differentiation, and XLA optimization, offering unparalleled speed and scalability for your vision workloads.
- Familiar API: Drawing inspiration from torchvision, Jaxvision provides a straightforward and intuitive API for instantiating models like ResNets, EfficientNets, and mor
- Modular and Flexible (NNX-powered): Thanks to Flax NNX, Jaxvision models are inherently modular and composable. NNX's explicit state management makes it easy to inspect, modify, and extend model components, giving you full control over your network architectures.
- Production-Ready: With JAX's robust infrastructure, models built with JAXvision are ideal for both research and production environments, providing a solid foundation for deploying high-performance computer vision applications.
  

Jaxvision currently includes implementations of widely-used architectures such as:
- *ResNet Family*: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, and ResNeXt-50.
- *EfficientNet Family*: fficientNet-B0 to B7, and EfficientNetV2-S, M, and L.

Each model is carefully implemented to reflect its original design, ensuring accurate and performant reproductions.


Using JAXvision is straightforward. You can easily instantiate a model and integrate it into your JAX training pipelines. For example, to create a ResNet-50:

```python
import jax
from flax import nnx
from models.resnet import resnet50


rngs = nnx.Rngs(0)
key = rngs.params()


model = resnet50(rngs=rngs, num_classes=1000)


dummy_input = jax.random.normal(key, (1, 224, 224, 3))


model.train()

output_train = model(dummy_input)
print(f"Output shape (training): {output_train.shape}")


model.eval()

output_eval = model(dummy_input)
print(f"Output shape (evaluation): {output_eval.shape}")
```
```python
Output shape (training):   (1, 1000)
Output shape (evaluation): (1, 1000)
```

JAXvision aims to be your go-to library for leveraging powerful vision models within the JAX ecosystem. We invite you to explore its capabilities and integrate it into your next JAX-powered computer vision project!
