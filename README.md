#Improving Privacy in Federated Learning Against Source Inference Attacks

### Core Idea
We propose a framework that combines homomorphic encryption (HE) and local differential privacy (LDP) for federated learning (FL). HE protects high-risk model parameters, while LDP secures less sensitive weights. Our approach aims to boost FL security against inference attacks without major performance drawbacks. The framework uses a selection strategy to identify vulnerable model parameters and applies HE only where necessary, enhancing efficiency. Additionally, LDP noise is added to low-risk parameters, and random shuffling improves privacy further.

### System Model of Federated Learning

A typical and simplified system model of federated learning is that a server is doing model aggregation for a bunch of clients that hold its own local data.

### System Model of Federated Learning

The threat model is based on the attacking method mentioned in the paper "Source Inference Attacks in Federated Learning" by Hu et al.

### Selective Encryption Strategy for High-risk Model Parameters

The selection strategy for these high-risk model parameters (i.e., high-risk proportion of the model's parameters) is critical. While traditional ideas may rely on the absolute values or the absolute changes (delta values) of the parameters, our framework introduces an innovative approach by evaluating the parameters' propensity, from the perspective of the adversary, to contribute to the Attack Success Rate (ASR) of SIA in the FL context, where the adversary acts as the honest but curious server. By assessing which parameters most significantly affect the loss function in the context of SIA, we can identify those parameters as high-risk.

### Environment Setup
```python
python --version
3.9.18
```
```pip
pip --version
23.3.1
```

You can run the following code to implement the source inference attacks. 

### Data generation
```python
python generate_synthetic.py
```

You can try different `--alpha` (data distribution), `--num_users`(number of parties), `--local_ep` (number of local epochs) to see how the attack performance changes. For `Synthetic` dataset, we set `--model=mlp`. For `MNIST` dataset, we set `--model=cnn`.

For `Synthetic` dataset, you can run the following code to generate it. But, a dataset has already been generated in the data/synthetic folder for usage.

### Pure Federated Learning (i.e., source inference attacks towards pure federated learning)

```python
python main_fed_pure.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```

### Federated Learning with Homomorphic Encryption

```python
python main_fed_combine_he.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```

### Federated Learning with Homomorphic Encryption and Differential Privacy

```python
python main_fed_combine_he_dp.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```

### Federated Learning with Homomorphic Encryption, Differential Privacy and Shuffler

```python
python main_fed_combine_he_dp_shuffle.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```
