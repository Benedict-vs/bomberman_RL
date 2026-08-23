"""
Breadth-First_Search (BFS) for pathfinding to the nearest coin in the game state.
This module provides functions to convert the game state into a feature representation suitable for Q-learning.
The main function `state_to_features` returns a tuple containing:
--  direction to the nearest coin (UP, RIGHT, DOWN, LEFT, or NONE if no coin is reachable)
-- status of the four neighboring fields (0 for free, 1 for blocked)

neighboring fields are checked for blockages (walls or crates) to inform the agent's decision making.
"""


import numpy as np
from collections import deque

def state_to_features(game_state: dict) -> tuple:
    """
    game state into tuple of features
    [coin_dir, up, right, down, left]
    """
    if game_state is None:
        return None

    field = game_state['field']
    _, _, _, (x, y) = game_state['self']
    coins = game_state['coins']


    coin_dir = get_coin_dir(x, y, field, coins) 

    #check neighbouring fields for blockages
    neighbours = get_neighbours(x, y, field) #ignore bombs for now, not relevant for coin collector agent

    #tuple is hashable, so can be used as a key in dict 
    return (coin_dir, *neighbours)


def get_coin_dir(x: int, y: int, field: np.array, coins: list) -> str:
    """
    Returns the direction to the nearest coin using BFS
    if no coin is reachable, returns 'NONE'
    """
    if not coins:
        return 'NONE'

    # queue for BFS, stores tuples of (current_position, first_step_direction)
    queue = deque([((x, y), 'NONE')])
    visited = {(x, y)}

    # define up , down, left, right directions with corresponding strings
    #image coords shape (x,y), so up (0,-1), down (0,1), left (-1,0), right (1,0)
    directions = [(0, -1, 'UP'), (1, 0, 'RIGHT'), (0, 1, 'DOWN'), (-1, 0, 'LEFT')]

    while queue:
        (cx, cy), first_step = queue.popleft()

         #check if coin reached
        if (cx, cy) in coins:
            return first_step

        # check all 4 directions
        for dx, dy, d_str in directions:
            nx, ny = cx + dx, cy + dy
            
            # check if new position is within bounds and not visited
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                #is field 0 or 1 (blocked or free)
                if field[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    next_first_step = d_str if first_step == 'NONE' else first_step
                    queue.append(((nx, ny), next_first_step))

    return 'NONE'


def get_neighbours(x: int, y: int, field: np.array) -> tuple:
    """
    checks neighbouring fields for blockages 
    returns a tuple of 4 values (up, right, down, left) where 0 = free, 1 = blocked
    """
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    neighbours = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        
        if (not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1])) or field[nx, ny] != 0:
            neighbours.append(1) 
        else:
            neighbours.append(0) 
            
    return tuple(neighbours)