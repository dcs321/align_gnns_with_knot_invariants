# Alignment of GNNs with Knot Invariants

This is the repository for the "L65: Geometric Deep Learning" mini-project on "Alignment of GNNs with Knot Invariants". We use the KnotInfo dataset [1] for knot structures and invariants.

## Installation

Create a Python environment with Conda:

```bash
conda create -n geom_knots python=3.11
```

Install the requirements inside the environment:

```bash
pip install -r requirements.txt
```

If using wandb, it is recommended to create a `.env` file for the variables:

```bash
export WANDB_API_KEY=<your_api_key_here>
export WANDB_PROJECT=<your_project_name_here>
```

And then source it:

```bash
source .env
```

## Using Different Node Features

Different node features can be chosen for the hypergraph representation of the knot, some of which can be embedded as well:

```bash
python run.py --wandb --target_invariant volume --node_feature_type numbers --embedding_used 
```

## Training Different Invariants

Models can be trained to predict different invariants from the graph structure, approached as either a regression or a classification task.

```bash
python run.py --wandb --target_invariant volume --regression_or_classification regression --criterion mse --node_feature_type laplacian 
```

or

```bash
python run.py --wandb --embedding_used --target_invariant alternating --regression_or_classification classification --criterion weighted_cross_entropy --node_feature_type numbers 
```

## Baseline

As a baseline, knot invariants can be predicted using a Neural Network (NN) [2] instead of HyperGCN [4]:

```bash
python run.py --wandb --embedding_used --target_invariant volume --node_feature_type numbers --model ffnn --data_type pd_notation --data_loader pd_notation 
```

## Misalignment Score

Misalignment score [3] can be a good proxy for GNN performance, both for classification and regression tasks:

```bash
python run.py --wandb --embedding_used --target_invariant determinant --node_feature_type numbers --compute_misaligment_score
```

## Laplacian and Periodic Features

By changing the number of used Laplacian eigenvectors as node features, or deploying periodic complex numbers or embeddings for this purpose, we can assess how much node feature information is required for good model performance.

```bash
python run.py --wandb --target_invariant volume --regression_or_classification regression --criterion mse --node_feature_type laplacian --number_of_laplacian 16
```

or

```bash
python run.py --wandb --target_invariant volume --regression_or_classification regression --criterion mse --node_feature_type complex_circular --number_of_period_in_complex_circular 12
```

## Connectivity Experiments

The knot hypergraph incidence matrix can be altered to measure the importance of topological structure during training and prediction.

```bash
python run.py --wandb --embedding_used --target_invariant three_colorability --regression_or_classification classification --criterion weighted_cross_entropy --node_feature_type numbers --connectivity random
```

## References

[1] Cha, J. C., & Livingston, C. (2011, May). Knotinfo: Table of knot invariants.

[2] Lindsay, A., & Ruehle, F. (2025). On the Learnability of Knot Invariants: Representation, Predictability, and Neural Similarity. *arXiv preprint arXiv:2502.12243*.

[3] Ayday, N., Sabanayagam, M., & Ghoshdastidar, D. (2025). Why does your graph neural network fail on some graphs? Insights from exact generalisation error. *arXiv preprint arXiv:2509.10337*.

[4] Bai, S., Zhang, F., & Torr, P. H. (2021). Hypergraph convolution and hypergraph attention. Pattern Recognition, 110, 107637.