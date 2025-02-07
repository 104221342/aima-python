from collections import deque
import sys
import numpy as np


# Read the input file
filename = sys.argv[1]
method = sys.argv[2]

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



   
        

def depth_first_route(grid, initial_state, goal_states):
    """
    Perform Depth-First Search (DFS) to find a path to the nearest goal state.
    Nodes are expanded according to the order: UP, LEFT, DOWN, RIGHT.
    Output format (if a goal is reached):
        filename method
        goal number_of_nodes
        path (sequence of directions)
    """
    stack = [(initial_state, [])]  # Each element: (current_state, path_taken)
    visited = set()
    nodes_expanded = 0

    # Movement order: UP, LEFT, DOWN, RIGHT
    moves = [(-1, 0, "UP"), (0, -1, "LEFT"), (1, 0, "DOWN"), (0, 1, "RIGHT")]
    
    while stack:
        current, path = stack.pop()
        nodes_expanded += 1

        if current in goal_states:
            print(f"{filename} {method}\n{current} {nodes_expanded}\n{' '.join(path)}")
            return

        if current in visited:
            continue
        visited.add(current)

        x, y = current
        # To ensure that the moves are expanded in the desired order,
        # we push them in reverse order onto the stack.
        for dx, dy, direction in reversed(moves):
            next_pos = (x + dx, y + dy)
            if 0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1]:
                if grid[next_pos] != 1 and next_pos not in visited:
                    stack.append((next_pos, path + [direction]))

    print(f"{filename} {method}\nNo goal is reachable; {nodes_expanded}")
    
def breadth_first_route(grid, initial_state, goal_states):
    """
    Perform Breadth-First Search (BFS) to find a path to the nearest goal state.
    Nodes are expanded in FIFO order, and the movement order is: UP, LEFT, DOWN, RIGHT.
    Output format (if a goal is reached):
        filename method
        goal number_of_nodes
        path (sequence of directions)
    """
    
    queue = deque()
    visited = set()
    
    queue.append((initial_state, []))
    visited.add(initial_state)
    nodes_expanded = 0

    # Movement order: UP, LEFT, DOWN, RIGHT
    moves = [(-1, 0, "UP"), (0, -1, "LEFT"), (1, 0, "DOWN"), (0, 1, "RIGHT")]
    
    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1

        if current in goal_states:
            print(f"{filename} {method}\n{current} {nodes_expanded}\n{' '.join(path)}")
            return

        x, y = current
        for dx, dy, direction in reversed(moves):
            next_pos = (x + dx, y + dy)
            if 0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1]:
                if grid[next_pos] != 1 and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [direction]))

    print(f"{filename} {method}\nNo goal is reachable; {nodes_expanded}")

    
'''def greedy_best_first_route(map):
    
def a_star_route(map):
    
def iterative_deepening_route(map):
    
def beam_route(map):'''

def main():
    # Ensure proper usage
    if len(sys.argv) < 3:
        print("Usage: python Assignment1.py <filename> <method>")
        sys.exit(1)


    # Parse the input file
    rows, cols, initial_state, goal_states, obstacles = parse_input(filename)

    # Create the grid
    grid = create_grid(rows, cols, initial_state, goal_states, obstacles)


    
    if method == "DFS":
        depth_first_route(grid, initial_state, set(goal_states)) 
    elif method == "BFS":
        breadth_first_route(grid, initial_state, set(goal_states))
    else:
        print("Unsupported search method.")
    
    

if __name__ == "__main__":
    main()