
#ANDROID APP SECURITY CLASSIFIER

### Load Necessary Libraries
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

"""###Data Preparation and Visualization"""

df = pd.read_csv('/content/data.csv')

df.head()

df.info()

df.shape

df.isnull().sum()

constants = [
    val for val in df.columns if len(df[val].fillna(0).unique()) == 1
]

constants

duplicate_variables = []
for i in range(0, len(df.columns)):
    orig = df.columns[i]

    for dupe in df.columns[i + 1:]:
        if df[orig].equals(df[dupe]):
            duplicate_variables.append(dupe)
            print(f'{orig} looks the same as {dupe}')

duplicate_variables

df.drop(['me.everything.badger.permission.BADGE_COUNT_READ'], axis=1, inplace=True)
df.shape

def visualize_class_distribution(df):
    print(df['Result'].value_counts())
    df['Result'].value_counts().plot(kind='bar')

visualize_class_distribution(df)

"""Our dataset is balanced so no need for oversampling"""

df.describe()

"""###Building, Training, and Evaluating the Model"""

X = torch.FloatTensor(df.drop('Result', axis=1).values)
y = torch.FloatTensor(df['Result'].values).view(-1, 1)

"""#### i. Building the model"""

class AndroidSecurityClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AndroidSecurityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Add a sigmoid activation function

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

"""#### ii. Training and Evaluation"""

def custom_confusion_matrix(y_true, y_pred):
    true_positive = torch.sum((y_true == 1) & (y_pred == 1)).item()
    false_positive = torch.sum((y_true == 0) & (y_pred == 1)).item()
    true_negative = torch.sum((y_true == 0) & (y_pred == 0)).item()
    false_negative = torch.sum((y_true == 1) & (y_pred == 0)).item()

    return torch.tensor([[true_negative, false_positive],
                         [false_negative, true_positive]])

def custom_accuracy_score(y_true, y_pred):
    correct_predictions = torch.sum(y_true == y_pred).item()
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

def custom_train_test_split(X, y, test_size=0.25, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)

    indices = torch.randperm(X.shape[0])
    num_test_samples = int(test_size * X.shape[0])

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def train_model_with_metrics(X_train, y_train, X_val, y_val, model, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_train = (model(X_train) >= 0.5).float()
            train_accuracy = custom_accuracy_score(y_train, y_pred_train)

        model.eval()
        with torch.no_grad():
            y_pred_val = (model(X_val) >= 0.5).float()
            val_loss = criterion(model(X_val), y_val)
            val_accuracy = custom_accuracy_score(y_val, y_pred_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Accuracy = {train_accuracy:.4f}, "
              f"Validation Accuracy = {val_accuracy:.4f}")

    return model, train_losses, val_losses, val_accuracies, train_accuracies



X_train, X_temp, y_train, y_temp = custom_train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = custom_train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

input_size = X_train.shape[1]
output_size = 1

learning_rates = [0.01, 0.1, 0.5]
num_epochs_list = [100, 150, 200]
hidden_sizes = [64, 128, 256]

best_accuracy = 0.0
best_lr = None
best_epochs = None
best_hidden_size = None
best_model = None

for lr in learning_rates:
    for num_epochs in num_epochs_list:
        for hidden_size in hidden_sizes:
            model = AndroidSecurityClassifier(input_size, hidden_size, output_size)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            y_train_float = y_train.float()
            y_val_float = y_val.float()

            trained_model, train_losses, val_losses, val_accuracies, train_accuracies = train_model_with_metrics(
            X_train, y_train_float, X_val, y_val_float, model, criterion, optimizer, num_epochs)


            print(f"\nLearning Rate: {lr}, Epochs: {num_epochs}, Hidden Size: {hidden_size}, "
                  f"Validation Accuracy: {val_accuracies[-1]:.4f}\n")

            if val_accuracies[-1] > best_accuracy:
                best_accuracy = val_accuracies[-1]
                best_lr = lr
                best_epochs = num_epochs
                best_hidden_size = hidden_size
                best_model = trained_model

print(f"Best Learning Rate: {best_lr}, Best Epochs: {best_epochs}, Best Hidden Size: {best_hidden_size}, "
      f"Best Validation Accuracy: {best_accuracy:.4f}")

if best_model is not None:
    best_model.eval()
    with torch.no_grad():
        y_pred_test = (best_model(X_test) >= 0.5).float()
        test_accuracy = custom_accuracy_score(y_test, y_pred_test)
        print(f"\nTest Accuracy with Best Model: {test_accuracy:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


else:
    print("No best model found. Check the hyperparameter search.")

"""###Results"""

y_test_tensor = torch.tensor(y_test.numpy())
y_pred_test_tensor = torch.tensor(y_pred_test.numpy())
cm = custom_confusion_matrix(y_test_tensor, y_pred_test_tensor)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

y_true_np = y_test.numpy()
y_pred_test_np = y_pred_test.numpy()

target_names = ["Benign", "Malicious"]

print("Classification Report:")
print(classification_report(y_true_np, y_pred_test_np, target_names=target_names))

fpr, tpr, threshold = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Android Security Classification')
plt.legend()
plt.grid()
plt.show()
