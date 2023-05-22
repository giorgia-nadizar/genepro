# genepro

## In brief

`genepro` is a Python library providing a baseline implementation of genetic programming, an evolutionary algorithm specialized to evolve programs.
This is a forked repository of the original one: <a href="https://github.com/marcovirgolin/genepro">genepro</a>.

Evolving programs are represented as trees.
The leaf nodes (also called *terminals*) of such trees represent some form of input, e.g., a feature for classification or regression, or a type of environmental observation for reinforcement learning.
The internal nodes represent possible atomic instructions, e.g., summation, subtraction, multiplication, division, but also if-then-else or similar programming constructs.

Genetic programming operates on a population of trees, typically initialized at random. 
Every iteration (called *generation*), promising trees undergo random modifications (e.g., forms of *crossover*, *mutation*, and *tuning*) that result in a population of offspring trees.
This new population is then used for the next generation.

### Full installation 
For a full installation, clone this repo locally, and make use of the file [requirements.txt](requirements.txt), as follows:
```
git clone https://github.com/giorgia-nadizar/genepro.git
cd genepro
pip3 install -r requirements.txt .
pip3 install -U .
```

### Wish to use conda?
A conda virtual enviroment can easily be set up with:
```
git clone https://github.com/giorgia-nadizar/genepro.git
cd genepro
conda env create
conda activate genepro
pip3 install -r requirements.txt .
pip3 install -U .
```

## Citation
If you use this software, please cite it with:
```
@software{Virgolin_genepro_2022,
  author = {Virgolin, Marco},
  month = {9},
  title = {{genepro}},
  url = {https://github.com/marcovirgolin/genepro},
  version = {0.1.0},
  year = {2022}
}
```
