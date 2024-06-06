# NeatClassification
This work explores the application of NeuroEvolution of Augmenting Topologies (NEAT) for binary classification tasks, utilizing the Matthews Correlation Coefficient (MCC) as the fitness function. We implement a NEAT-based classifier and evaluate its performance on a real-world dataset. The results demonstrate the efficacy of NEAT in evolving neural network architectures that achieve high MCC scores, indicating robust classification performance.
## Installation
To install the required packages please run the following :
```bash
pip install -r requirements.txt
```
## usage 
To use the app check the neat_config file to choose mainly your inputs and outputs under the DefaultGenome property as follows:
```yaml
[DefaultGenome]
feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 6
num_outputs             = 1
```
other requirements can also be tweaked following the official guide [here](https://neat-python.readthedocs.io/en/latest/config_file.html) on the official documentation.

To run simply type : 
```bash
python test.py
``` 