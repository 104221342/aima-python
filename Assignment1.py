import sys
import numpy as np

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
# Read the input file
filename = sys.argv[1]
# method = sys.argv[2]

def parse_input(filename):
    """Parse the input file to extract grid size, start state, goals, and obstacles."""
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    # Grid dimensions
    grid_dims = eval(lines[0])  # [rows, cols]
    rows, cols = grid_dims

    # Initial state
    initial_state = eval(lines[1])  # (row, col)

    # Goal states
    goal_states = [eval(goal) for goal in lines[2].split('|')]  # [(row1, col1), (row2, col2)]

    # Obstacles
    obstacles = [eval(obstacle) for obstacle in lines[3:]]  # [(row, col, width, height), ...]

    return rows, cols, initial_state, goal_states, obstacles

def create_grid(rows, cols, initial_state, goal_states, obstacles):
    """Create a grid and mark the initial state, goals, and obstacles."""
    # Initialize the grid with empty cells (0)
    grid = np.zeros((rows, cols), dtype=int)

    # Mark the initial state (-1)
    grid[initial_state[1],initial_state[0]] = -1

    # Mark the goal states (2)
    for goal in goal_states:
        grid[goal[1],goal[0]] = 2

    # Mark obstacles (1)
    for obstacle in obstacles:
        top, left, width, height = obstacle
        grid[left:left+height, top: top+width] = 1

    return grid

def print_grid(grid):
    """Utility to print the grid in a readable format."""
    print("Grid:")
    for row in grid:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    # Ensure proper usage
    if len(sys.argv) < 2:
        print("Usage: python Assignment1.py <input_file>")
        sys.exit(1)


    # Parse the input file
    rows, cols, initial_state, goal_states, obstacles = parse_input(filename)

    # Create the grid
    grid = create_grid(rows, cols, initial_state, goal_states, obstacles)

    # Print the grid
    print_grid(grid)

   
        

'''def depth_first_route(map):
    # Start at the first node
    current_node = list(map.keys())[0]
    
def breath_first_route(map):
    
def greedy_best_first_route(map):
    
def a_star_route(map):
    
def iterative_deepening_route(map):
    
def beam_route(map):'''
