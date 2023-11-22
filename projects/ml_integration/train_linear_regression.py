import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from surrealml import SurMlFile

squarefoot = np.array([1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200], dtype=np.float32)
num_floors = np.array([1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3], dtype=np.float32)
house_price = np.array([200000, 230000, 280000, 320000, 350000, 380000, 420000, 470000, 500000, 520000], dtype=np.float32)

squarefoot_mean = squarefoot.mean()
squarefoot_std = squarefoot.std()
num_floors_mean = num_floors.mean()
num_floors_std = num_floors.std()
house_price_mean = house_price.mean()
house_price_std = house_price.std()

# Normalize the data (optional, but recommended for better convergence)
squarefoot = (squarefoot - squarefoot.mean()) / squarefoot.std()
num_floors = (num_floors - num_floors.mean()) / num_floors.std()
house_price = (house_price - house_price.mean()) / house_price.std()

# Convert numpy arrays to PyTorch tensors
squarefoot_tensor = torch.from_numpy(squarefoot)
num_floors_tensor = torch.from_numpy(num_floors)
house_price_tensor = torch.from_numpy(house_price)

# Combine features into a single tensor
X = torch.stack([squarefoot_tensor, num_floors_tensor], dim=1)


# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2 input features, 1 output

    def forward(self, x):
        return self.linear(x)


# Initialize the model
model = LinearRegressionModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred.squeeze(), house_price_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

test_squarefoot = torch.tensor([2800, 3200], dtype=torch.float32)
test_num_floors = torch.tensor([2.5, 3], dtype=torch.float32)
test_inputs = torch.stack([test_squarefoot, test_num_floors], dim=1)
test_inputs = torch.tensor([2800, 3], dtype=torch.float32)

# Test the model
with torch.no_grad():
    predicted_prices = model(test_inputs)
    predicted_prices = predicted_prices.squeeze().numpy()
    print("Predicted Prices:", predicted_prices)


file = SurMlFile(model=model, name="Prediction", inputs=test_inputs)

# # export to ONNX and save file
# torch.onnx.export(model, test_inputs, "./linear_test.onnx")

file.add_description("This model predicts the price of a house based on its square footage and number of floors.")
file.add_version("0.0.1")

file.add_column("squarefoot")
file.add_column("num_floors")
file.add_output("house_price", "z_score", house_price_mean, house_price_std)

file.add_normaliser("squarefoot", "z_score", squarefoot_mean, squarefoot_std)
file.add_normaliser("num_floors", "z_score", num_floors_mean, num_floors_std)
file.save("./linear_test.surml")

new_file = SurMlFile.load("./linear_test.surml")
print(new_file)

# print(new_file.raw_compute([1.0, 2.0], (1, 2)))
print(new_file.raw_compute([1.0, 2.0]))


print(new_file.buffered_compute({
    "squarefoot": 3200.0,
    "num_floors": 2.0
}))


# m = model.load_state_dict()
d = model.state_dict()
print(d)


if __name__ == "__main__":
    pass
