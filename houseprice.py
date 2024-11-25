"""
pip install mlflow
2.5.2 >= you need torch >= 1.9.0 (as of nov 25 2024)

# start server command
mlflow server --host 127.0.0.1 --port [port-var-below]

# can access through browser by doing https://localhost:[port-var-below]

this jist of how to use it then is 
wrap your training loop with a thing to log it
basically this is the code you need to insert and this is where we insert it

You can also create different expirements to keep track of things. There is a photo in the 
repo on the reccomended way to do this
"""

port = 8080

import sys
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

argc = len(sys.argv)
if (argc < 2):
    print("Please provide a experiment name")
    exit()

experiment_name = sys.argv[1]


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        # First layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        # Second layer
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        # Output layer
        out = self.fc3(out)
        return out
    
#***** load and clean data *****
from datasets import load_dataset

ds = load_dataset("ttd22/house-price")

inputs = []
answers = []

#Get desired items in datasett
for sample in ds['train']:
    fireplaces = sample['Fireplaces']
    year_built = sample['YearBuilt']
    price = sample['SalePrice']
    
    inputs.append([fireplaces, year_built])
    answers.append(price)

inputs = torch.tensor(inputs, dtype=torch.float32)
answers = torch.tensor(answers, dtype=torch.float32)

input_size = 2
hidden_size1 = 100
hidden_size2 = 100
output_size = 1

model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

# loss func and optim
lossFunc = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# *****Training time! *****

#http://127.0.0.1:[port-var-below] - make sure server is started on this port
mlflow.set_tracking_uri(f'http://127.0.0.1:{port}')

# Provide an Experiment description that will appear in the UI
experiment_description = (
    "This is a test to see how creating experiemnts works"
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": experiment_name,
    "team": "me-and-carter",
    "mlflow.note.content": experiment_description,
}

# Create the Experiment, providing a unique name
if mlflow.get_experiment_by_name(experiment_name) is None:
    produce_apples_experiment = mlflow.create_experiment(
        name=experiment_name, tags=experiment_tags
    )


experiment = mlflow.set_experiment(experiment_name)

num_epochs = 1000
mlflow.pytorch.autolog()

with mlflow.start_run(run_name=f"{datetime.datetime.now()}"):
    #log hyperparameters
    mlflow.log_param("input_size", input_size)
    mlflow.log_param("hidden_size1", hidden_size1)
    mlflow.log_param("hidden_size2", hidden_size2)
    mlflow.log_param("output_size", output_size)
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_param("learning_rate", 0.01)

    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        predictions = model(inputs)
        loss = lossFunc(predictions, answers)

        # Zero gradients (using optimizer)
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            #log loss
            mlflow.log_metric("loss", loss.item(), step=epoch)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    #log model
    mlflow.pytorch.log_model(model, "model")


#Testing time
ds = load_dataset("animonte/train_house_price")

model.eval()

#disable gradient tracking to speed up performance 
with torch.no_grad():
    #Get desired items in datasett
    for sample in ds['train']:
        fireplaces = sample['Fireplaces']
        year_built = sample['YearBuilt']
        price = sample['SalePrice']

        inputs = []
        inputs.append([fireplaces, year_built])
        
        inputs = torch.tensor(inputs, dtype=torch.float32)
        print(f"Input: {inputs.tolist()} | Prediction: ${int(model(inputs).item())} | Answer: ${price}")
    
