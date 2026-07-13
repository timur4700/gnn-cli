## Project Description

GNN-CLI is a framework designed to simplify the development, training, and evaluation of Graph Neural Network (GNN)-based models, primarily for protein–ligand binding affinity prediction. The framework can also be applied to general molecular property prediction tasks.

## Project Status

🚧 **Active development**

The project is currently under active development. It includes core functionality for GNN model training and evaluation, but some CLI commands are still limited and known issues remain.

## Current Features

- Ligand-based GNN training pipeline
- Support for Message Passing Neural Network (MPNN) architectures:
    - Graph Convolutional Networks (GCN)
    - Graph Isomorphism Networks (GIN)
    - Graph Attention Networks (GAT)
- Configurable training and graph preprocessing pipelines
- Model registry system for managing architectures
- Command-line interface
- Performance evaluation and metric calculation

## Planned Improvements

- Addition of Equivariant Graph Neural Networks (EGNN)
- Extension to protein–ligand interaction models
- More flexible model construction and configuration
- Multiprocessed graph preparation
- Support for additional molecular formats:
    - `.mol2`
    - `.pdb`
- Improved documentation
- Bug fixes and expanded testing

## Technical Stack

- Python
- PyTorch
- PyTorch Geometric
- scikit-learn
- RDKit
- NumPy
- SciPy
- JSON-based configuration
- Conda environment management


## Installation
### 1. Install Conda

Install Miniconda or Anaconda if it is not already installed.

### 2. Create the Conda Environment

Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml -y




---

# Installation

## 1. Install Conda

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

Verify installation:

```bash
conda --version
```

---

## 2. Create Conda Environment

The required environment is provided through:

```
environment.yml
```

Create the environment:

```bash
conda env create -f environment.yml -y
```

The environment name can be modified inside `environment.yml`:

```yaml
name: gnn-env
```

Change `gnn-env` to your preferred environment name.

Activate the environment:

```bash
conda activate gnn-env
```

---

# Example Data

The repository includes example molecular datasets under the `example_data` directory.

These files can be used to test the complete workflow without requiring external datasets.

Example structure:

```
example_data/
│
├── train_molecules.sdf
└── test_molecules.sdf
```

The example dataset contains ligand molecules with the target property encoded directly in the SDF files.

---

# Usage

The framework is controlled through a command-line interface.

## Create a New Project

To create a new project:

```bash
python main.py mkproj
```

A project directory and configuration files will be generated.

---

## Creating a Project with Arguments

You can provide all required information directly:

```bash
python main.py mkproj \
    -n project_name \
    --train_path train_molecules.sdf \
    --test_path test_molecules.sdf \
    --property IC50
```

Arguments:

| Argument | Description |
|---|---|
| `-n` | Project name |
| `--train_path` | Path to training SDF file |
| `--test_path` | Path to testing SDF file |
| `--property` | Target molecular property |

---

## Important Note About Molecular Properties

At the current project stage, the target property must already exist inside the SDF files.

Example:

```
train_molecules.sdf
    |
    └── IC50 property

test_molecules.sdf
    |
    └── IC50 property
```

If required arguments are not provided, the program will interactively ask for:

- Project name
- Training molecule path
- Testing molecule path
- Target property name

---

# Graph Preparation Configuration

After project creation, graph preprocessing parameters can be configured.

---

## Atomic Features

Available atomic features:

```
Atomic Number
Atom Charge
Aromaticity
Number of Hydrogens
Atom Hybridization
Atomic Mass
Atom Valence
Atom Degree
```

Feature encoding methods:

### Standard Encoding

Uses continuous or categorical numerical representation.

### One-Hot Encoding

Uses binary feature vectors for discrete values.

> Note: One-hot encoding currently contains known issues and is under development.

---

# Bond Features

The following bond features can be enabled or disabled:

- Bond Type
- Stereo Type
- Aromaticity Status
- Interatomic Distance

---

# Graph Representation

Two graph construction methods are supported:

## Chemical Topology

Represents molecular connectivity based on chemical bonds.

Uses molecular topology information from RDKit.

---

## Spatial Topology

Uses three-dimensional atomic coordinates.

Connections are generated based on spatial relationships between atoms.

---

# Model Configuration

After graph preparation, model architecture and hyperparameters can be configured.

## Available Models

Currently supported GNN architectures:

### Graph Convolutional Network (GCN)

A convolution-based graph neural network.

### Graph Isomorphism Network (GIN)

A powerful message passing architecture designed to distinguish graph structures.

### Graph Attention Network (GAT)

Uses attention mechanisms to weight neighboring atoms.

---

# Model Hyperparameters

Available configuration options include:

- Hidden dimension size
- Number of layers
- Optimizer
- Learning rate
- Weight decay
- Random seed

Current limitation:

```
Dropout rate is fixed to 0.1
```

and cannot currently be modified through configuration.

---

# Training Configuration

Training parameters can be configured through the project settings.

Available options:

- Number of epochs
- Batch size
- Learning rate
- Weight decay
- Optimizer selection
- Validation split
- Early stopping
- Device selection:
    - CPU
    - CUDA
- Training log saving (not available now)

---

# Training a Model

After completing project creation and configuration, the model can be trained using the `train` command.

To start training, provide the project name using the `-n` argument:

```bash
python main.py train -n project_name
```

The selected project will use the previously configured:

- Graph preparation settings
- Model architecture
- Model hyperparameters
- Training parameters

During training, the framework will automatically:

1. Load the project configuration.
2. Prepare molecular graph data.
3. Initialize the selected GNN model.
4. Train the model according to the configured training parameters.
5. Evaluate the trained model on the test molecules.

After training, model performance will be evaluated using the selected evaluation metrics, such as:

- Pearson correlation coefficient (`Rp`)
- Spearman correlation coefficient (`Rs`)
- Coefficient of determination (`R²`)
- Root Mean Square Error (`RMSE`)
- Mean Absolute Error (`MAE`)

Example:

```bash
python main.py train -n ligand-based
```

---

# Deleting a Project

To delete an existing project, use the `delete` command:

```bash
python main.py delete
```

After executing the command, a project selector will appear.

Select the project you want to remove and confirm the deletion.

The deletion process removes the selected project from:

- The project registry
- The project directory
- Stored configuration files

Example workflow:

```
$ python main.py delete

Select project to delete:

> ligand_based_1
  ligand_based_2
  ligand_based_3

---

# Example Workflow

A typical workflow:

```
1. Create project
        |
        v
2. Prepare molecular graphs
        |
        v
3. Configure atom/bond features
        |
        v
4. Select GNN model
        |
        v
5. Configure training parameters
        |
        v
6. Train model
        |
        v
7. Evaluate performance
```

---

# Technical Stack

The project is developed using:

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- SciPy
- Conda

---

# Current Limitations

The current development version has several limitations:

- Only ligand-based models are currently implemented.
- Protein-ligand interaction graphs are not yet supported.
- Some CLI functionality is still under development.
- One-hot feature encoding requires further validation.
- Automated testing coverage is limited.

---

# Contributing

This project is currently under active development.
Suggestions, bug reports, and improvements are welcome.

---