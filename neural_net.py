import torch
import torch.nn as nn
import torch.nn.functional as F
from cube import Cube, FACES

# Color encoding
color_map = {'W': 0, 'R': 1, 'B': 2, 'O': 3, 'G': 4, 'Y': 5}
num_colors = len(color_map) 

MODEL_FILE_PATH = "cube_solver_model.pth"

# Maps moves to indices
move_map = {
    "U": 0, "U'": 1,
    "D": 2, "D'": 3,
    "L": 4, "L'": 5,
    "R": 6, "R'": 7,
    "F": 8, "F'": 9,
    "B": 10, "B'": 11
}
# Maps indices back to moves
index_to_move = {v: k for k, v in move_map.items()}
num_moves = len(move_map)

# Encodes a cube state as a 324-dimensional one-hot vector
def encode_cube(cube):
    onehot_vector = []
    
    for face_name in FACES:
        face = cube[face_name]
        for row in face:
            for color in row:
                onehot = [0] * num_colors
                onehot[color_map[color]] = 1
                onehot_vector.extend(onehot)
    return torch.tensor(onehot_vector, dtype=torch.float32)

# Encodes a move as an integer
def encode_move(move):
    return move_map[move]

# Decodes an index to a move string
def decode_move(index):
    return index_to_move[index]

# Neural network for solving the Rubik's Cube
class CubeSolverNet(nn.Module):
    def __init__(self):
        super(CubeSolverNet, self).__init__()
        self.fc1 = nn.Linear(54 * num_colors, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_moves)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # raw logits
