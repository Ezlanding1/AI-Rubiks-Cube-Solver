import random
import copy
from collections import deque 
import torch
import torch.nn.functional as F
from cube import Cube 
from neural_net import encode_cube, decode_move, CubeSolverNet, MODEL_FILE_PATH

# Initialize data and load the model
device = torch.device("cpu")
model = CubeSolverNet().to(device)
model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device))

# Solve the cube using a neural network model, without heuristic search
def solve_cube(cube: Cube, max_moves=5000):

    def predict_next_move(cube: Cube):
        model.eval()
        x = encode_cube(cube.faces).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            predicted_index = torch.argmax(logits, dim=1).item()
            return decode_move(predicted_index)
    
    i = 0
    solution = []
    while not cube.is_solved():
        predicted_move = predict_next_move(cube)
        solution.append(predicted_move)
        cube.move([predicted_move])
        
        if i >= max_moves:
            return None
        i += 1

    return solution
    

def main(scrambles=5, cube: Cube = None, scramble_sequence=None):
    # Create and scramble a cube
    if not cube:
        cube = Cube()
        scramble_sequence = Cube.inverse_sequence(cube.scramble(scrambles=scrambles))
    else:
        assert scramble_sequence is not None
        
    print("Scramble:", scramble_sequence)

    print("Solving Cube...")

    solution = solve_cube(cube)

    return solution

if __name__ == '__main__':
    if solution := main():
        print("Solution Found:", solution)
    else:
        print("No Solution Found")