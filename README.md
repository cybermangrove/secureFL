Python version:
```python
python --version
3.9.18
```
```python
pip --version
23.3.1
```


You can run the following code to implement the source inference attacks. For `Synthetic` dataset, you can run the following code to generate it. A current dataset was already generated in the data/synthetic folder.
```python
python generate_synthetic.py
```

You can try different `--alpha` (data distribution), `--num_users`(number of parties), `--local_ep` (number of local epochs) to see how the attack performance changes. For `Synthetic` dataset, we set `--model=mlp`. For `MNIST` dataset, we set `--model=cnn`.

```python
python main_fed_pure.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```
```python
python main_fed_combine_he.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```
```python
python main_fed_combine_he_dp.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```
```python
python main_fed_combine_he_dp_shuffle.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```

