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

# Solve the cube using beam search 
def solve_cube_with_beam_search(cube, max_moves=20, beam_width=5, max_itr=10000):
    
    model.eval() # Set model to evaluation mode

    q = deque([(copy.deepcopy(cube), [])])
    
    # Convert cube faces to a hashable tuple
    def get_hashable_state(cube):
        return tuple(tuple(tuple(row) for row in face) for face in cube.faces.values())

    # Store visited states to avoid loops
    visited_states = {get_hashable_state(cube)}
    
    solved_state_hash = get_hashable_state(Cube()) 

    itr = 0
    while q:
        current_cube, path = q.popleft()

        if len(path) > max_moves:
            continue # Prune paths that are too long

        if get_hashable_state(current_cube) == solved_state_hash:
            print(f"Solved in {len(path)} moves!")
            return path

        # Get model's prediction for the current state
        with torch.no_grad():
            x = encode_cube(current_cube.faces).unsqueeze(0)
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)

            # Get top 'beam_width' moves
            _, top_indices = torch.topk(probabilities, beam_width)
            
            # Explore top predicted moves
            for i in range(beam_width):
                predicted_move_idx = top_indices[0, i].item()
                predicted_move_str = decode_move(predicted_move_idx)

                next_cube = copy.deepcopy(current_cube)
                next_cube.move([predicted_move_str])

                next_state_hash = get_hashable_state(next_cube)

                if next_state_hash not in visited_states:
                    visited_states.add(next_state_hash)
                    q.append((next_cube, path + [predicted_move_str]))

        if itr > max_itr:
            break
        itr += 1
                    
    return None

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
    

def main(scrambles=5, solve_type='beam', cube: Cube = None, scramble_sequence=None):
    # Create and scramble a cube
    if not cube:
        cube = Cube()
        scramble_sequence = Cube.inverse_sequence(cube.scramble(scrambles=scrambles))
    else:
        assert scramble_sequence is not None
        
    print("Scramble:", scramble_sequence)

    print("Solving Cube...")

    if solve_type == 'beam':
        solution = solve_cube_with_beam_search(cube, beam_width=3)
    else:
        solution = solve_cube(cube)

    return solution

if __name__ == '__main__':
    if solution := main():
        print("Solution Found:", solution)
    else:
        print("No Solution Found")