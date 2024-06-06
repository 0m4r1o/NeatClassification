import neat
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = 'dataset.csv'
dataset = pd.read_csv(file_path)

# Preprocess the dataset
# Drop non-numeric and irrelevant columns
dataset = dataset.drop(columns=['ID', 'Agency', 'Destination', 'Product Name', 'Distribution Channel'])

# Handle missing values (e.g., filling NaNs with the mean)
dataset = dataset.fillna(dataset.mean())

# Convert categorical columns to numeric
le = LabelEncoder()
categorical_cols = ['Agency Type', 'Gender']
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col].astype(str))

# Separate features and target
X = dataset.drop(columns=['Claim']).values
y = dataset['Claim'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        y_pred_train = []
        for xi in X_train:
            output = net.activate(xi)
            y_pred_train.append(output[0] > 0.5)
        
        # Calculate MCC for the training predictions
        mcc = matthews_corrcoef(y_train, y_pred_train)*100
        # print(f'MCC = {mcc} ')
        genome.fitness = mcc

# Set up the NEAT configuration
config_path = "neat_config"  # Make sure to have a neat_config file in the same directory
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run for up to 100 generations.
winner = p.run(eval_genomes, 100)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Evaluate the best genome on the test data
net = neat.nn.FeedForwardNetwork.create(winner, config)
y_pred_test = []
for xi in X_test:
    output = net.activate(xi)
    y_pred_test.append(output[0] > 0.5)

mcc_test = matthews_corrcoef(y_test, y_pred_test)
print(f"Test MCC: {mcc_test:.3f}")
