"""
Artificial Neural Network using 3 FC layers and BN and Dropout for regularization.
Note: Hyper-parameter tuning is not implemented here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn import model_selection
import copy
from models.model_helper_fxns import *


class CustomDataset(Dataset):
    """
    Custom Dataset using torch format. Reads in a dataframe of values of both the features and target.
    A list of features and a string target are supplied as initialization inputs.
    """
    def __init__(self, df, features, target):
        df = df.reset_index()
        self.df_x = df[features]
        self.df_y = df[target]

    def __len__(self):
        return self.df_x.shape[0]

    def __getitem__(self, item):
        inputs = torch.Tensor(list(self.df_x.iloc[item]))
        outputs = torch.Tensor([self.df_y.iloc[item]])
        return inputs, outputs


class CustomNetwork(nn.Module):
    """
    Custom ANN architecture using the torch.nn.Module format.
    Includes 3 FC Layers with batchnorm and dropout for regularization.
    """
    def __init__(self, features):
        super().__init__()
        self.lin1 = nn.Linear(len(features), 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.lin2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.lin3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.lin4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(self.lin3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.lin4(x)
        return x


def run_ANN(df_train_orig, df_test, features, target, random_state=42, feature_grouping='All Features'):
    """
    Implementation of an ANN model. Does not perform hyper-parameter search.
    Performs training and evaluation of model using following hyper-parameters:
        Adam optimizer with lr = 0.001 and weight_decay = 0
        n_epochs = 1000
        Standard network architecture defined above.
    Metrics are reported as RMSE and %Error in both table and graph.

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param df_test: Pandas dataframe of the testing data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :param random_state: Int representing the random state in the train/val split
    :param feature_grouping: String representing name of feature set. Default = 'All Features". Meant
           to serve as identifier on graphed results as we vary input feature sets
    :return: List in the following format: [train rmse, train percent error, test rmse, test percent error]
    """
    # Define Datasets and Dataloaders
    df_train, df_val = model_selection.train_test_split(df_train_orig, test_size=0.25, random_state=random_state)

    train_dataset = CustomDataset(df_train, features, target)
    val_dataset = CustomDataset(df_val, features, target)
    test_dataset = CustomDataset(df_test, features, target)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Define model
    model = CustomNetwork(features)
    model = model.to(device)

    # Define loss function and optimizers
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    n_epochs = 1000
    train_loss_cumulative = []
    val_loss_cumulative = []
    best_weights = None
    best_val_score = float('inf')

    # Model Training and Validation Loop for N_Epochs
    for epoch in tqdm(range(n_epochs)):
        print(f'Epoch {epoch}')
        train_loss = []
        val_loss = []

        # Training: Predictions, loss, gradient optimization
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            preds = model(x)
            loss = loss_fn(preds, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item() / y.shape[0])
        train_loss_cumulative.append(np.average(train_loss))
        print(f'Train Loss: {train_loss_cumulative[-1]}')

        # Validation: Predictions, loss
        model.eval()
        for batch in val_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            val_loss.append(loss.item() / y.shape[0])
        val_loss_cumulative.append(np.average(val_loss))
        print(f'Validation Loss: {val_loss_cumulative[-1]}')

        # Save best model weights based on validation score
        if val_loss_cumulative[-1] < best_val_score:
            best_val_score = val_loss_cumulative[-1]
            best_weights = copy.deepcopy(model.state_dict())

    # Plot training curve
    x_values = [i for i in range(n_epochs)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x_values, train_loss_cumulative, color='blue', label='train')
    ax.plot(x_values, val_loss_cumulative, color='red', label='validation')
    ax.legend()
    ax.set(xlabel=f'Epoch', ylabel=f'Loss (MSE)', title='ANN Training Curve')
    plt.show()

    # Load best model weights based on validation scores
    model.load_state_dict(best_weights)
    model.eval()

    # Calculate loss for entire test dataset
    test_loss = 0
    for batch in test_loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        test_loss += loss.item()

    test_loss = test_loss / df_test.shape[0]
    print(f'Test MSE Loss: {test_loss}')

    # Calculating Metrics (MSE and Percent Error)
    train_metrics_loader = DataLoader(train_dataset, batch_size=df_train.shape[0], shuffle=False)
    batch = next(iter(train_metrics_loader))
    train_metrics = calc_metrics(model, df_train, features, target, type='ANN', model_input=batch[0].to(device))
    test_metrics_loader = DataLoader(test_dataset, batch_size=df_test.shape[0], shuffle=False)
    batch = next(iter(test_metrics_loader))
    test_metrics = calc_metrics(model, df_test, features, target, type='ANN', model_input=batch[0].to(device))

    # Plot observed vs. predicted cycle life
    title = f'Artificial Neural Network \n Feature Set: {feature_grouping}'
    plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title)

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
