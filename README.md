# Diffusion-Jump-GNNs
Diffusion-Jump GNNs: Homophiliation via Learnable Metric Filters
## Dependencies

Conda environment
```
conda create --name <env> --file requirements.txt
```

or

```
conda env create -f conda_graphy_environment.yml
conda activate graphy
```
## Code organization
* `splits/`: splits that we used, taking from GEO-GCN repository.
* `main.py`: script with inline arguments for running the experiments.
* `main_scalable.py`: script with inline arguments for running large experiments.
* `models.py`: script with our proposed architecture.
* `pump.py`: implementation of the proposed **pump**.
* `utils.py`: extra functions used for the experiments.
## Run experiments
```python
python main.py --dataset texas --hidden_channels 64 --dropout 0.2 --lr 0.03 --n_layers 20 --epochs 700 --cuda cpu
python main.py --dataset wisconsin --hidden_channels 64 --dropout 0.5 --lr 0.03 --n_layers 5 --epochs 700 --cuda cpu
python main.py --dataset cornell --hidden_channels 128 --dropout 0.5 --lr 0.03 --n_layers 5 --wd 0.001 --epochs 700 --cuda cpu
```
