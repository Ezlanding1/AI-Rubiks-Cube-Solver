import torch
import torch.nn as nn
import torch.optim as optim
from cube import Cube
from torch.utils.data import Dataset
import copy
from torch.utils.data import DataLoader
from neural_net import encode_cube, encode_move, CubeSolverNet, MODEL_FILE_PATH

# Train the model
def train_on_cube(cube, solution, model, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for move in solution:
        # Encode cube state
        x = encode_cube(cube).unsqueeze(0) 
        y = torch.tensor([encode_move(move)])

        # Forward, loss, backward
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        cube.move([move])

    return total_loss / len(solution)

# Dataset for Rubik's Cube states and moves
class RubiksCubeDataset(Dataset):
    def __init__(self, num_samples=1000, scrambles=20):
        self.samples = []

        for _ in range(num_samples):

            cube = Cube()
            solution = cube.scramble(scrambles=scrambles)

            # Collect (state, move) pairs 
            for move in solution:
                state = copy.deepcopy(cube.faces)  
                self.samples.append((state, move))
                cube.move([move])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, move = self.samples[idx]
        x = encode_cube(state)               
        y = encode_move(move)             
        return x, y

# Train the model for one epoch
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    import time

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CubeSolverNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Cross Entropy loss is used for classification of categorical data
    criterion = nn.CrossEntropyLoss()

    #model.load_state_dict(torch.load(MODEL_FILE_PATH))

    # Dataset and DataLoader
    dataset = RubiksCubeDataset(num_samples=1000, scrambles=20)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Training loop
    num_epochs = 750
    for epoch in range(num_epochs):
        start = time.time()
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        duration = time.time() - start
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Time: {duration:.2f}s")

    # Save model
    torch.save(model.state_dict(), MODEL_FILE_PATH)


if __name__ == "__main__":
    main()

