from cube import Cube

# Convert cube colors to console colors
console_color = {
	'Y' : '\033[33m■ \033[0m',
	'B' : '\033[34m■ \033[0m',
	'R' : '\033[31m■ \033[0m',
	'G' : '\033[32m■ \033[0m',
	'O' : '\033[38;5;214m■ \033[0m',
	'W' : '\033[37m■ \033[0m',
    None : '  '
}

def merge_faces(face1, face2) -> str:
    return '\n'.join([f"{a} {b}" for a, b in zip(face1.split('\n'), face2.split('\n'))])[:-1]

# Render a single face of the cube
def render_face(face) -> str:
    result = ''
    
    for row in face:
        result += str.join('', [ console_color[color] for color in row ]) + '\n'

    return result

# Render the entire cube
def render_cube(cube: Cube):
    result = ''

    PADDING = render_face([([ None ] * 3) for _ in range(3)])

    up = render_face(cube.faces['U'])
    front = render_face(cube.faces['F'])
    left = render_face(cube.faces['L'])
    right = render_face(cube.faces['R'])
    back = render_face(cube.faces['B'])
    down = render_face(cube.faces['D'])

    result += merge_faces(PADDING, up)
    result += merge_faces(merge_faces(merge_faces(left, front), right), back)
    result += merge_faces(PADDING, down)
    return result