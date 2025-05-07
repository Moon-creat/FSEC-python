import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from fsec import FSEC
from measures.cluster_acc import cluster_acc
from measures.clustering_measure import clustering_measure
from measures.rand_index import rand_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo for FSEC with merged MNIST dataset')
    parser.add_argument('--nearest',   type=int, default=10,  help='Number of nearest anchors')
    parser.add_argument('--base',      type=int, default=40, help='Number of base clusterings')
    parser.add_argument('--anchor',    type=int, default=12, help='Number of anchors (optional)')
    parser.add_argument('--subset',    type=int, default=70000, help='Total number of samples from combined dataset')
    args = parser.parse_args()

    # 1. 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1))
    ])

    # 2. 加载训练集和测试集
    train_ds = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 3. 合并数据集
    combined_ds = ConcatDataset([train_ds, test_ds])

    # 4. 从合并数据集中提取子集
    loader = DataLoader(combined_ds, batch_size=args.subset, shuffle=True)
    X_tensor, Y_tensor = next(iter(loader))  # 取前 subset 样本

    # 5. 执行 FSEC
    labels, anchors, B, acc, nmi, ari = FSEC(
        X_tensor, Y_tensor,
        num_nearest_anchor=args.nearest,
        num_base=args.base,
        exponent_p=args.anchor
    )

    # 6. 打印结果
    print(f"Total samples: {args.subset}")
    print(f"Accuracy: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")