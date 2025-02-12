from collections import deque
import heapq
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


def is_valid_move(pos, grid):
    """
    Check whether moving to the position 'pos' (row, col) is valid.
      - It must be within the grid boundaries.
      - The cell must not be an obstacle (i.e., grid value != 1).
    """
    x, y = pos
    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
        return False
    if grid[x, y] == 1:  # Obstacle check
        return False
    return True

def heuristic(state, goal_states):
    """
    Compute the Manhattan distance from 'state' to the nearest goal.
    """
    x, y = state
    return min(abs(x - gx) + abs(y - gy) for gx, gy in goal_states)
   
        

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
            if is_valid_move(next_pos,grid) and next_pos not in visited:
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
            # The move is valid if it's within bounds and not an obstacle.
            if is_valid_move(next_pos, grid) and next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [direction]))

    print(f"{filename} {method}\nNo goal is reachable; {nodes_expanded}")

    
def greedy_best_first_route(grid, initial_state, goal_states):
    """
    Greedy Best-First Search (GBFS) to find a path to one of the goal states.
    Uses the Manhattan distance heuristic to always expand the node closest to a goal.
    Tie-breaking is handled by a counter (earlier nodes expanded first) and the
    movement order (UP, LEFT, DOWN, RIGHT).
    """
    
    
    nodes_expanded = 0
    counter = 0  # Tie-breaker for nodes with equal heuristic value
    
    # Priority queue: each entry is (heuristic, counter, current_state, path)
    heap = []
    start_h = heuristic(initial_state, goal_states)
    heapq.heappush(heap, (start_h, counter, initial_state, []))
    
    visited = set()
    visited.add(initial_state)
    
    # Movement order: UP, LEFT, DOWN, RIGHT
    moves = [(-1, 0, "UP"), (0, -1, "LEFT"), (1, 0, "DOWN"), (0, 1, "RIGHT")]
    
    while heap:
        h_value, _, current, path = heapq.heappop(heap)
        nodes_expanded += 1

        if current in goal_states:
            print(f"{filename} {method}\n{current} {nodes_expanded}\n{' '.join(path)}")
            return

        x, y = current
        for dx, dy, direction in moves:
            next_state = (x + dx, y + dy)
            if is_valid_move(next_state, grid) and next_state not in visited:
                visited.add(next_state)
                counter += 1
                h_next = heuristic(next_state, goal_states)
                heapq.heappush(heap, (h_next, counter, next_state, path + [direction]))
    
    print(f"{filename} {method}\nNo goal is reachable; {nodes_expanded}")

    
def a_star_route(grid, initial_state, goal_states):
    """
    A* Search to find a path to one of the goal states.
    Uses the cost function f = g + h where g is the cost so far and h is the Manhattan
    distance heuristic to the nearest goal. Tie-breaking is handled by a counter.
    The movement order is: UP, LEFT, DOWN, RIGHT.
    """

    nodes_expanded = 0
    counter = 0  # Tie-breaker counter
    
    # Priority queue: each entry is (f, counter, current_state, g, path)
    start_h = heuristic(initial_state, goal_states)
    start_f = start_h  # g is 0 initially, so f = h
    heap = []
    heapq.heappush(heap, (start_f, counter, initial_state, 0, []))
    
    # Record the best cost to a node so far
    g_values = {initial_state: 0}
    
    moves = [(-1, 0, "UP"), (0, -1, "LEFT"), (1, 0, "DOWN"), (0, 1, "RIGHT")]
    
    while heap:
        f, _, current, g, path = heapq.heappop(heap)
        nodes_expanded += 1
        
        if current in goal_states:
            print(f"{filename} {method}\n{current} {nodes_expanded}\n{' '.join(path)}")
            return
        
        x, y = current
        for dx, dy, direction in moves:
            neighbor = (x + dx, y + dy)
            if is_valid_move(neighbor, grid):
                new_g = g + 1
                # Only consider this neighbor if we haven't visited it with a lower cost
                if neighbor in g_values and new_g >= g_values[neighbor]:
                    continue
                g_values[neighbor] = new_g
                new_h = heuristic(neighbor, goal_states)
                new_f = new_g + new_h
                counter += 1
                heapq.heappush(heap, (new_f, counter, neighbor, new_g, path + [direction]))
    
    print(f"{filename} {method}\nNo goal is reachable; {nodes_expanded}")

def depth_limited_search(grid, state, goal_states, depth_limit, path, visited):
    """
    Depth-limited search (DLS) for use in Iterative Deepening Depth-First Search (IDDFS).
    """
    if state in goal_states:
        return path, True  # Found a goal
    
    if depth_limit == 0:
        return None, False  # Depth limit reached

    x, y = state
    moves = [(-1, 0, "UP"), (0, -1, "LEFT"), (1, 0, "DOWN"), (0, 1, "RIGHT")]
    
    cutoff_occurred = False
    for dx, dy, direction in moves:
        next_state = (x + dx, y + dy)
        if is_valid_move(next_state, grid) and next_state not in visited:
            visited.add(next_state)
            result, found = depth_limited_search(grid, next_state, goal_states, depth_limit - 1, path + [direction], visited)
            if found:
                return result, True
            if result is None:
                cutoff_occurred = True  # If a deeper level could be useful

    return None, cutoff_occurred
    
def iterative_deepening_route(grid, initial_state, goal_states):
    """
    Iterative Deepening Depth-First Search (IDDFS).
    """
     
    depth = 0
    nodes_expanded = 0

    while True:
        visited = {initial_state}
        path, found = depth_limited_search(grid, initial_state, goal_states, depth, [], visited)
        nodes_expanded += len(visited)  # Count visited nodes at each depth level
        
        if found:
            print(f"{filename} {method}\n{initial_state} {nodes_expanded}\n{' '.join(path)}")
            return
        elif path is None:  # No goal found and no deeper search possible
            print(f"{filename} {method}\nNo goal is reachable; {nodes_expanded}")
            return
        
        depth += 1  # Increase depth limit for the next iteration


def beam_route(grid, initial_state, goal_states, beam_width=2):
    """
    Beam Search Algorithm:
    - Expands only 'beam_width' best nodes per level.
    - Uses Manhattan distance as the heuristic.
    """
    
    # Priority queue (heuristic value, node, path)
    frontier = [(heuristic(initial_state, goal_states), initial_state, [])]
    
    visited = set()
    nodes_expanded = 0
    
    moves = [(-1, 0, "UP"), (0, -1, "LEFT"), (1, 0, "DOWN"), (0, 1, "RIGHT")]

    while frontier:
        next_frontier = []
        nodes_expanded += len(frontier)
        
        # Expand only the best 'beam_width' nodes at this depth
        for _, (x, y), path in sorted(frontier)[:beam_width]:
            if (x, y) in goal_states:
                print(f"{filename} {method}\n{x, y} {nodes_expanded}\n{' '.join(path)}")
                return

            visited.add((x, y))

            for dx, dy, direction in moves:
                next_state = (x + dx, y + dy)
                if is_valid_move(next_state, grid) and next_state not in visited:
                    next_frontier.append((heuristic(next_state, goal_states), next_state, path + [direction]))

        # Update frontier with the new level's best nodes
        frontier = sorted(next_frontier)[:beam_width]

    print(f"{filename} {method}\nNo goal is reachable; {nodes_expanded}")

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
        depth_first_route(grid, initial_state, goal_states) 
    elif method == "BFS":
        breadth_first_route(grid, initial_state, goal_states)
    elif method == "GBFS":
        greedy_best_first_route(grid, initial_state, goal_states)
    elif method == "AS":
        a_star_route(grid,initial_state, goal_states)
    elif method == "CUS1":
        iterative_deepening_route(grid, initial_state, goal_states)
    elif method == "CUS2":
        beam_route(grid, initial_state, goal_states, 2)
    else:
        print("Unsupported search method.")
    
    

if __name__ == "__main__":
    main()