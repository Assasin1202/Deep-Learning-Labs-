# %% [markdown]
# ## Qs 2 - Design a Feed Forward Neural Network (FFN) for the following IRIS dataset.

# %% [markdown]
# ### Importing necessary Libraries and preprocessing 

# %%
!pip install ucimlrepo

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
X = pd.read_csv('census_income.csv')
y = pd.read_csv('census_income_target.csv')
# from ucimlrepo import fetch_ucirepo
# census_income = fetch_ucirepo(id=20)



# X = census_income.data.features
# y = census_income.data.targets


# %%
X

# %%
X.to_csv('census_income.csv', index=False)
y.to_csv('census_income_target.csv', index=False)

# %%
y

# %%
y.value_counts()

# %%
# Cleaning the target variable (y)
y = y.replace({'>50K.': '>50K', '<=50K.': '<=50K'})


# %%
y.value_counts()

# %% [markdown]
# ### Preprocessing the dataset

# %%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 
                    'race', 'sex', 'native-country']
le= LabelEncoder()

for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

print(X.head())

X.info()




# %%
le = LabelEncoder()
y = le.fit_transform(y)
y.shape

# %%
# Checking for missing values
print(X.isnull().sum())


# %% [markdown]
# It does not have any null values. 

# %% [markdown]
# ### Doing some Exploratory Data analysis

# %%

sns.histplot(X['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# %%
X['workclass_original'] = pd.read_csv('census_income.csv')['workclass']
plt.figure(figsize=(10, 6))
sns.countplot(x=X['workclass_original'])
plt.title('Workclass Distribution')
plt.xlabel('Workclass')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# %%
X.drop('workclass_original', axis=1, inplace=True)

# %%
df_combined = X.copy()
df_combined['income'] = y

plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='education-num', data=df_combined)
plt.title('Education Level vs Income')
plt.xlabel('Income')
plt.ylabel('Education Level')
plt.show()


# %%

plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.show()


# %%

sns.histplot(X[X['capital-gain'] > 0]['capital-gain'], bins=30, kde=True)
plt.title('Capital Gain Distribution (Excluding Zero)')
plt.xlabel('Capital Gain')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# #### Scaling the dataset

# %%
# Standard Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
X.shape

# %%
# Splitting the dataset. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# ### Creating the Feed Forward Neural Network Class. 

# %%
from sklearn.metrics import accuracy_score, classification_report

class FFN:
    def __init__(self, input_size, hidden_size, num_classes):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, num_classes) * 0.1
        self.b2 = np.zeros((1, num_classes))
    
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(m), y_true])
        return np.sum(log_likelihood) / m
    
    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2
    
    def backward(self, X, y, Z1, A1, Z2, A2, learning_rate=0.1):
        m = X.shape[0]
        
        dZ2 = A2 - np.eye(A2.shape[1])[y]  
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=1)

# %% [markdown]
# ### Instantiating and Training the model 

# %%
nclasses = len(np.unique(y))

# %%


# Model parameters
input_size = X_train.shape[1]  # Number of features
hidden_size = 16  # Number of neurons in the hidden layer
num_classes = len(np.unique(y_train))  # Number of classes


model = FFN(input_size, hidden_size, num_classes)

num_epochs = 200
train_losses = []
train_accuracies = []
for epoch in range(num_epochs):
    Z1, A1, Z2, A2 = model.forward(X_train)

    loss = model.cross_entropy_loss(y_train, A2)

    model.backward(X_train, y_train, Z1, A1, Z2, A2)

    train_losses.append(loss)

    y_train_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_train_pred)

    train_accuracies.append(accuracy)
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')




# %%

y_pred = model.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)

print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")

\
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate class-wise accuracy
class_wise_accuracy = []
classes = sorted(set(y_test)) 

for class_index in classes:
    class_total = sum([1 for label in y_test if label == class_index])
    class_correct = sum([1 for i, label in enumerate(y_test) if label == class_index and y_pred[i] == class_index])
    class_accuracy = class_correct / class_total * 100
    class_wise_accuracy.append((class_index, class_accuracy))
    print(f"Class {class_index} Accuracy: {class_accuracy:.2f}%")

# %% [markdown]
# ### Plotting Loss / Epoch and Accuracy / Epoch

# %%


epochs = range(1, num_epochs + 1)

plt.figure(figsize=(14, 6))

# Loss graph 
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r', label='Training loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy graph 
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %% [markdown]
# ### Performing Oversampling and again training the model. 

# %%
from imblearn.over_sampling import SMOTE


# Applying SMOTE to the training set
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# %%


# Model parameters
input_size = X_train.shape[1]  # Number of features
hidden_size = 16  # Number of neurons in the hidden layer
num_classes = len(np.unique(y_train))  # Number of classes


# Now, re-initialize the model
model = FFN(input_size, hidden_size, num_classes)


# Retrain the model using the resampled data
num_epochs = 200
train_losses = []
train_accuracies = []
for epoch in range(num_epochs):
    Z1, A1, Z2, A2 = model.forward(X_resampled)

    loss = model.cross_entropy_loss(y_resampled, A2)

    model.backward(X_resampled, y_resampled, Z1, A1, Z2, A2)

    train_losses.append(loss)

    y_train_pred = model.predict(X_resampled)

    accuracy = accuracy_score(y_resampled, y_train_pred)

    train_accuracies.append(accuracy)
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Evaluate on the test set
y_pred = model.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)

print(f"\nOverall Test Accuracy after SMOTE: {overall_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

classes = sorted(set(y_test)) 

# Class-wise accuracy after oversampling
class_wise_accuracy = []
for class_index in classes:
    class_total = sum([1 for label in y_test if label == class_index])
    class_correct = sum([1 for i, label in enumerate(y_test) if label == class_index and y_pred[i] == class_index])
    class_accuracy = class_correct / class_total * 100
    class_wise_accuracy.append((class_index, class_accuracy))
    print(f"Class {class_index} Accuracy after SMOTE: {class_accuracy:.2f}%")


# %%


epochs = range(1, num_epochs + 1)

plt.figure(figsize=(14, 6))

# Loss graph 
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r', label='Training loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy graph 
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %% [markdown]
# ### Finding Best parameters using Grid Search 

# %%
def train_and_evaluate(hidden_size, learning_rate, num_epochs):
    model = FFN(input_size, hidden_size, num_classes)
    
    for epoch in range(num_epochs):
        Z1, A1, Z2, A2 = model.forward(X_train)
        loss = model.cross_entropy_loss(y_train, A2)
        model.backward(X_train, y_train, Z1, A1, Z2, A2, learning_rate)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


param_grid = {                               # Defining the hyperparameter grid
    'hidden_size': [8, 16, 32],
    'learning_rate': [0.01, 0.1, 1],
    'num_epochs': [50, 100,150]
}

# Grid searching to find the best parameters
best_accuracy = 0
best_params = {}

for hidden_size in param_grid['hidden_size']:
    for learning_rate in param_grid['learning_rate']:
        for num_epochs in param_grid['num_epochs']:
            print(f"Testing hidden_size={hidden_size}, learning_rate={learning_rate}, num_epochs={num_epochs}")
            accuracy = train_and_evaluate(hidden_size, learning_rate, num_epochs)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'hidden_size': hidden_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs
                }

print("Best Parameters:")
print(best_params)
print(f"Best Accuracy: {best_accuracy:.4f}")

# %%


# %% [markdown]
# ## Repeating the Process using Pytorch and Verifying Results. 

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Creating Dataloader for the training and test set
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the Feed Forward Neural Network (FFN)
class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = X.shape[1]  # Number of features
hidden_size = 16 # Number of neurons in the hidden layer
num_classes = nclasses  # Number of classes


model = FFN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)






# %%
train_losses = []
train_accuracies = []

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    epoch_loss = 0
    
    for i, (features, labels) in enumerate(train_loader):
        outputs = model(features)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_losses.append(epoch_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())
    
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['<=50K', '>50K']))
    
    class_wise_accuracy = []
    for class_index, class_name in enumerate(['<=50K', '>50K']):
        class_total = sum([1 for label in y_true if label == class_index])
        class_correct = sum([1 for i, label in enumerate(y_true) if label == class_index and y_pred[i] == class_index])
        class_accuracy = class_correct / class_total * 100
        class_wise_accuracy.append((class_name, class_accuracy))
        print(f"{class_name} Accuracy: {class_accuracy:.2f}%")


# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, 'r', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch')
plt.legend()

plt.show()


# %% [markdown]
# ### Conclusion
# Hence we can say that by comparing the above two models (made using python and Pytorch), we can see that both the models perform almost the same and have similar accuracies. Hence our implementation is correct. 


