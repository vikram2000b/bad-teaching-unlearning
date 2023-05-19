# Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher

This code is official implementation of [Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher.](https://arxiv.org/abs/2205.08096)

By Vikram S Chundawat, Ayush K Tarun, Murari Mandal, Mohan Kankanhalli 

Unlearning Methods Supported:
1. Unlearning using an Incompetent Teacher (presented in this work)
2. Amnesiac Unlearning [presented as Unlearning in Graves, Nagisetty, and Ganesh 2021 (https://ojs.aaai.org/index.php/AAAI/article/view/17371/17178)]
3. UNSIR [presented as in Tarun et al. 2021 (https://arxiv.org/pdf/2111.08947.pdf)]

Models supported:
1. ResNet18
2. Vision Transformer (ViT) (has some issues with older pytorch versions)
3. AllCNN (comment transforms.Resize in datasets.py to work)

Datasets supported:
1. CIFAR-Super20 (mentioned in the paper)
2. CIFAR-10 (works similar to CIFAR-Super20, can be imported from torchvision.datasets and used)
3. CIFAR-100 (works similar to CIFAR-Super20, can be imported from torchvision.datasets and used)

Metrics Supported:
1. Performance (Accuracy)
2. Activation Distance
3. Membership Inference Attack
