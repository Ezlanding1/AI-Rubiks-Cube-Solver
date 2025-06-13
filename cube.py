import random
import numpy

FACES = [ 'U', 'F', 'R', 'B', 'L', 'D' ]
COLORS = [ 'Y', 'B', 'R', 'G', 'O', 'W' ]

OPPOSITE_FACE = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U', 'F': 'B', 'B': 'F'}
INVERSE_MOVE="'"


class Cube:
    
    def __init__(self):
        
        self.faces = {
            face :
            numpy.array([ 
                ([ color for _ in range(3)]) for _ in range(3) 
            ])
            for face, color in zip(FACES, COLORS)
        }

        self.adj = {
            'F' : [ ('U', 'R3b'), ('R', 'C1b'), ('D', 'R1'), ('L', 'C3') ],
            'B' : [ ('U', 'R1'), ('L', 'C1b'), ('D', 'R3b'), ('R', 'C3') ],

            'U' : [ ('B', 'R1'), ('R', 'R1'), ('F', 'R1'), ('L', 'R1') ],
            'D' : [ ('F', 'R3b'), ('R', 'R3b'), ('B', 'R3b'), ('L', 'R3b') ],
            
            'R' : [ ('U', 'C3'), ('B', 'C1b'), ('D', 'C3'), ('F', 'C3') ],
            'L' : [ ('U', 'C1b'), ('F', 'C1b'), ('D', 'C1b'), ('B', 'C3') ]
        }

    def get_row_or_col(self, fidx, idx):

        face = self.faces[fidx]
        
        if idx[0] == 'R':
            result = face[int(idx[1])-1]
        else:
            result = face[:, int(idx[1])-1]

        if idx[-1] == 'b':
            return numpy.flip(result)
        return result

    def move(self, sequence):
        for move in sequence:
            face, inverse = move[0], move[-1] == INVERSE_MOVE

            self.faces[face] = numpy.rot90(self.faces[face], k= -1 if not inverse else 1)
            edges = [self.get_row_or_col(*self.adj[face][i]) for i in range(3, -1, -1)]

            edges = list(reversed((edges))) if inverse else edges
            edges.append(edges[0].copy())


            for i in range(len(edges)-1):
                edges[i][0], edges[i][1], edges[i][2] = edges[i+1][0], edges[i+1][1], edges[i+1][2]

    def generate_scramble(scrambles=20):
        sequence = []
        lastOps = []

        for _ in range(scrambles):
            while True:
                op = random.choice(FACES)
                if op in lastOps: continue
                elif OPPOSITE_FACE[op] in lastOps: lastOps.append(op)
                else: lastOps = [op]
                break
            inverse = random.choice(['', INVERSE_MOVE])
            sequence.append(f'{op}{inverse}')

        return sequence

    def inverse_sequence(sequence):
        return list(reversed(
            [
                move[:-1] if move[-1] == INVERSE_MOVE else (move + INVERSE_MOVE) for move in sequence
            ]
        ))
    
    def scramble(self, scrambles=20):
        sequence = []

        sequence = Cube.generate_scramble(scrambles=scrambles)

        self.move(sequence)

        return Cube.inverse_sequence(sequence)
        
    def is_solved(self):
        reference_cube = Cube()

        for face in FACES:
            if not numpy.all(self.faces[face] == reference_cube.faces[face]):
                return False
        return True

    
if __name__ == '__main__':
    cube = Cube()
    print(cube)
    solution = cube.scramble()
    print(cube)
    cube.move(solution)
    print(cube)
