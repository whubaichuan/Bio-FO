# Efficient On-Device Machine Learning with a Biologically-Plausible Forward-Only Algorithm (Accepted by [MLSys 2025](https://mlsys.org/Conferences/2025))

This repository provides the reproducible code for all the reported results in the paper **Efficient On-Device Machine Learning with a Biologically-Plausible Forward-Only Algorithm**.

![Bio-FO](./img/main.png)


## 1. Bio-FO
In the directory of `BioFO`, the codes for the state-of-the-art forward-only algorithms are provided, including DRTP<sup>[1]</sup>, PEPITA<sup>[2]</sup>, FF<sup>[3]</sup>, and Bio-FO on MNIST, CIFAR-10, CIFAR-100, [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), [MIT-BIH](https://www.physionet.org/content/mitdb/1.0.0/) datastes. 
Taking MNIST as an example, the codes are shown as follows:
- MNIST-BP: ``python BioFO/MNIST/MNIST_BP/Main.py ``
- MNIST-DRTP: ``python BioFO/MNIST/MNIST_DRTP/main.py ``
- MNIST-PEPITA: ``python BioFO/MNIST/MNIST_PEPITA/pepita_MNIST.py``
- MNIST-FF: ``python BioFO/FF/main.py``; the parameters `dataset` and `class_num` in ``FF/config.yaml`` need to be changed accordingly for other datasets.
- MNIST-BioFO: ``python BioFO/MNIST/MNIST_BioFO/Main.py``


The following figure shows the error (%) evaluation for Backpropagation, DRTP, PEPITA, FF, and our proposed Bio-FO. (The solid line reports the mean over five independent runs, the shaded area indicates the standard deviation.)

![result](./img/results.png)

The following table shows the energy consumption (Wh) for the state-of-the-art DRTP, PEPITA, FF, and Bio-FO, where energy consumption is the product of training time and the electrical power of the NVIDIA Jetson Nano.

![result](./img/energy.png)

<!--The following figure shows the test error (\%) and fitting the test error by the *plateau equation for learning curves*<sup>[2]</sup>.
![result](./img/convergence.png)-->


## 2. Extension to LC & CNN

In the directory of `LC_CNN`, the codes for Bio-FO and BP with LC and CNN are provided. 

Taking CIFAR-10 as an example, the codes are shown as follows:
- CIFAR-10-Bio-FO-LC: ``python LC_CNN/LC/BioFO.py ``
- CIFAR-10-Bio-FO-CNN: ``python LC_CNN/CNN/BioFO.py ``
- CIFAR-10-BP-LC: ``python LC_CNN/LC/BP.py ``
- CIFAR-10-BP-CNN: ``python LC_CNN/LC/BP.py``

> the parameters `dataset: 1-mnist,2-cifar10,3-cifar100` need to be changed accordingly.


## 3. Extension to [mini-ImageNet](https://paperswithcode.com/dataset/mini-imagenet)

In the directory of `mini-ImageNet`, the codes for DRTP, PEPITA, FF, CaFo<sup>[4]</sup>, Bio-FO, and BP are provided, which are shown as follows:
- mini-ImageNet-DRTP: ``python mini-ImageNet/DRTP/main.py ``
- mini-ImageNet-PEPITA: ``python mini-ImageNet/DRTP/pepita.py ``
- mini-ImageNet-FF: ``python mini-ImageNet/FF/main.py ``
- mini-ImageNet-CaFo: ``python mini-ImageNet/CaFo/train.py ``
- mini-ImageNet-Bio-FO: ``python mini-ImageNet/BioFO.py ``
- mini-ImageNet-BP: ``python mini-ImageNet/BP.py ``


[1] Frenkel, Charlotte, Martin Lefebvre, and David Bol. "Learning without feedback: Fixed random learning signals allow for feedforward training of deep neural networks." Frontiers in neuroscience 15 (2021): 629892.

[2] Dellaferrera, Giorgia, and Gabriel Kreiman. "Error-driven input modulation: solving the credit assignment problem without a backward pass." International Conference on Machine Learning. PMLR, 2022.

[3] Hinton, Geoffrey. "The forward-forward algorithm: Some preliminary investigations." arXiv preprint arXiv:2212.13345 (2022).


[4] Zhao, Gongpei, et al. "The cascaded forward algorithm for neural network training." Pattern Recognition 161 (2025): 111292.