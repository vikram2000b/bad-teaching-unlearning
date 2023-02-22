# Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher

This code is official implementation of [Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher.](https://arxiv.org/abs/2205.08096)

By Vikram S Chundawat, Ayush K Tarun, Murari Mandal, Mohan Kankanhalli 

Models supported:
1. ResNet18
2. Vision Transformer (ViT) (has some issues with older pytorch versions)
3. AllCNN (comment transforms.Resize in datasets.py to work)

Datasets supported:
1. CIFAR-Super20 (mentioned in the paper)
2. CIFAR-10 (works similar to CIFAR-Super20, can be imported from torchvision.datasets and used)
3. CIFAR-100 (works similar to CIFAR-Super20, can be imported from torchvision.datasets and used)