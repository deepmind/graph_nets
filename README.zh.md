![Graph Nets DeepMind shortest path](https://github.com/deepmind/graph_nets/raw/master/images/graph-nets-deepmind-shortest-path0.gif)

# Graph Nets 库 (Graph Nets library)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[Graph Nets](https://github.com/deepmind/graph_nets) 是 DeepMind 推出的用于在 TensorFlow 和 Sonnet 中构建图网络（Graph Networks）的官方库。

如有任何意见或问题，请联系 graph-nets@google.com。

#### 什么是图网络（Graph Networks）？

图网络以图（Graph）作为输入，并返回图作为输出。输入图包含边级（*E*）、节点级（*V*）和全局级（**u**）属性。输出图具有相同的拓扑结构，但属性经过了更新。图网络属于更广泛的“图神经网络（Graph Neural Networks）”（Scarselli et al., 2009）家族的一部分。

若要深入了解图网络，请参阅我们的 arXiv 论文：[Relational inductive biases, deep learning, and graph networks（关系归纳偏置、深度学习与图网络）](https://arxiv.org/abs/1806.01261)。

![Graph network](https://github.com/deepmind/graph_nets/raw/master/images/graph-network.png)

## 安装 (Installation)

Graph Nets 库可以通过 pip 安装。

该安装方式兼容 Linux/Mac OS X 系统，以及 Python 2.7 和 3.4+。

本库同时支持 CPU 和 GPU 版本的 TensorFlow，但为了支持这种灵活性，它没有将 TensorFlow 列为强制依赖项，因此如果您尚未安装 TensorFlow，需要单独进行安装。

若要安装 Graph Nets 库并搭配 TensorFlow 1 和 Sonnet 1 使用，请运行：

(CPU)
```shell
$ pip install graph_nets "tensorflow>=1.15,<2" "dm-sonnet<2" "tensorflow_probability<0.9"
```

(GPU)
```shell
$ pip install graph_nets "tensorflow_gpu>=1.15,<2" "dm-sonnet<2" "tensorflow_probability<0.9"
```

若要安装 Graph Nets 库并搭配 TensorFlow 2 和 Sonnet 2 使用，请运行：

(CPU)
```shell
$ pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```

(GPU)
```shell
$ pip install graph_nets "tensorflow_gpu>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```

最新版本的库需要 TensorFlow >=1.15。如需兼容更早版本的 TensorFlow，请安装 Graph Nets 库的 v1.0.4 版本。

## 使用示例 (Usage example)

以下代码构建了一个简单的图网络模块并将其连接到数据：

```python
import graph_nets as gn
import sonnet as snt

# 提供您自己的函数来生成图结构数据。
input_graphs = get_graphs()

# 创建图网络。
graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))

# 将输入图传递给图网络，并返回输出图。
output_graphs = graph_net_module(input_graphs)
```

## 演示 Jupyter 笔记本 (Demo Jupyter notebooks)

本库包含多个演示示例，展示了如何创建、操作和训练图网络，以便在最短路径查找任务、排序任务以及物理预测任务中对图结构数据进行推理。每个演示均使用相同的图网络架构，突显了该方法的灵活性。

### 在浏览器的 [Colaboratory](https://colab.research.google.com) 中体验演示

无需在本地安装任何环境，您即可通过云端 Colaboratory 后端直接在浏览器（甚至手机）中运行这些演示。请点击下方的演示链接，并按照笔记本中的说明操作。

----------------

#### [在浏览器中运行“最短路径演示 (shortest path demo)”](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/shortest_path.ipynb)

“最短路径演示”会创建随机图，并训练一个图网络来标记任意两个节点之间最短路径上的节点与边。经过一系列消息传递（message-passing）步骤后（如每一步的图表所示），模型会不断优化其对最短路径的预测结果。

![Shortest path](https://github.com/deepmind/graph_nets/raw/master/images/shortest-path.png)

----------------

#### [在浏览器中运行“排序演示 (sort demo)”](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/sort.ipynb)  [(运行 TF2 版本)](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos_tf2/sort.ipynb)

“排序演示”会创建随机数列表，并训练一个图网络对该列表进行排序。经过一系列消息传递步骤后，模型能够准确预测哪些元素（图中的列）紧接着其他元素（行）出现。

![Sort](https://github.com/deepmind/graph_nets/raw/master/images/sort.png)

----------------

#### [在浏览器中运行“物理演示 (physics demo)”](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/physics.ipynb)

“物理演示”会创建随机的弹簧-质点（mass-spring）物理系统，并训练一个图网络来预测系统在下一个时间步的状态。模型的下一步预测结果可以作为输入重新反馈给模型，从而生成未来轨迹的推演（rollout）。下方的每个子图展示了 50 个时间步内真实与预测的弹簧-质点系统状态。这与 Battaglia et al. (2016) 提出的“交互网络（interaction networks）”中的模型和实验相类似。

![Physics](https://github.com/deepmind/graph_nets/raw/master/images/physics.png)

----------------

#### [在浏览器中运行“图网络基础演示 (graph nets basics demo)”](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/graph_nets_basics.ipynb)  [(运行 TF2 版本)](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos_tf2/graph_nets_basics.ipynb)

“图网络基础演示”是一份循序渐进的教程，包含了有关如何创建和操作图、如何将图输入图网络以及如何构建自定义图网络模块的分步示例。

----------------

### 在本地机器上运行演示

要安装必要的依赖项，请运行：

```shell
$ pip install jupyter matplotlib scipy
```

要体验演示示例，请运行：

```shell
$ cd <path-to-graph-nets-library>/demos
$ jupyter notebook
```
随后通过 Jupyter Notebook 界面打开演示文件即可。


## 其他图神经网络库 (Other graph neural network libraries)

欢迎了解以下高质量的开源图神经网络库：

* [jraph](https://github.com/deepmind/jraph)：DeepMind 针对 [JAX](https://github.com/google/jax) 打造的 GNNs/GraphNets 库。

* [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)：请参阅 [MetaLayer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer) 以了解与我们的 Graph Nets 接口相对应的实现。

* [Deep Graph Library (DGL)](https://github.com/dmlc/dgl)。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月19日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
