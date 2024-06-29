import pandas as pd

from sklearn.datasets import load_boston

dataset = load_boston()
dataFrame = pd.DataFrame(dataset["data"])
dataFrame.columns = dataset["feature_names"]
dataFrame["target"] = dataset["target"]

print(dataFrame.head())

import torch
import torch.nn as nn

from torch.optim.adam import Adam

model = nn.Sequential(
    nn.Linear(13, 100),
    nn.ReLU(),
    nn.Linear(100,1)
)

X = dataFrame.iloc[:, :13].values
Y = dataFrame["target"].values

batch_size = 100
learning_rate = 0.001

optim = Adam(model.parameters(), lr=learning_rate)

for epoch in range(200):


    for i in range(len(X)//batch_size):
        start = i*batch_size
        end = start + batch_size

x = torch.FloatTensor(X[start:end])
y = torch.FloatTensor(Y[start:end])

optim.zero_grad()
preds = model(x)
loss = nn.MSELoss()(preds, y)
loss.backward()
optim.step()

if epoch % 20 == 0:
    print(f"epoch{epoch} loss:{loss.item()}")

    prediction = model(torch.FloatTensor(X[0, :13]))
real = Y[0]
print(f"prediction:{prediction.item()} real:{real}")
