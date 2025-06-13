import os
from time import sleep
from cube import Cube
import solve_cube

SCRAMBLE_COUNT = 6

def clear_output(notebook):
    if notebook:
        from IPython.display import clear_output
        clear_output(wait=True)
    else:
        CLEAR_COMMAND = 'cls' if os.name == 'nt' else 'clear'
        os.system(CLEAR_COMMAND)

def main(notebook=False):

    cube = Cube()

    print(cube)

    print("Generating Scramble...")
    scramble_r = Cube.generate_scramble(scrambles=SCRAMBLE_COUNT)
    scramble = []

    for move in scramble_r:
        clear_output(notebook)
        print("Generating Scramble...")
        scramble.append(move)
        print("Scramble: ", scramble)
        sleep(0.5)
        
    while scramble:
        clear_output(notebook)
        print("Applying Scramble...")
        move = scramble.pop(0)
        print("Scramble: ", scramble)
        cube.move([move])
        print(cube)
        sleep(0.5)


    i_cube = Cube()
    i_cube.move(scramble_r)
    
    solution = solve_cube.main(scrambles=SCRAMBLE_COUNT, cube=i_cube, scramble_sequence=scramble_r)

    if not solution:
        clear_output(notebook)
        print("No Solution Found")
        return

    while solution:
        clear_output(notebook)
        print("Solving...")
        move = solution.pop(0)
        cube.move([move])
        print(cube)
        sleep(0.5)

    clear_output(notebook)
    print("Solved!")
    print(cube)

if __name__ == "__main__":
    main()
