#Features für Step 1: Münzen sammeln ohne Kisten, Gegner oder Bomben

from collections import deque


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]

DIRECTIONS = (
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (0, 1),   # DOWN
    (-1, 0),  # LEFT
)

NO_COIN = 0
COIN_UP = 1
COIN_RIGHT = 2
COIN_DOWN = 3
COIN_LEFT = 4

FREE = 0
BLOCKED = 1


def direction_to_nearest_coin(field, start, coins):
    #Bestimmt den ersten Schritt eines kürzesten Weges zur nächsten Münze.

    targets = set(coins)
    start = tuple(start)

    if not targets:
        return NO_COIN

    if start in targets:
        return NO_COIN

    queue = deque()
    visited = {start}

    for direction_code, (dx, dy) in enumerate(DIRECTIONS, start=1):
        neighbour = (start[0] + dx, start[1] + dy)

        if field[neighbour] == 0:
            visited.add(neighbour)
            queue.append((neighbour, direction_code))

    while queue:
        position, first_direction = queue.popleft()

        if position in targets:
            return first_direction

        for dx, dy in DIRECTIONS:
            neighbour = (
                position[0] + dx,
                position[1] + dy,
            )

            if neighbour in visited:
                continue

            if field[neighbour] != 0:
                continue

            visited.add(neighbour)
            queue.append((neighbour, first_direction))

    return NO_COIN


def local_obstacles(field, position):
    #Codiert die vier Nachbarfelder als frei oder blockiert

    neighbour_codes = []

    for dx, dy in DIRECTIONS:
        neighbour = (
            position[0] + dx,
            position[1] + dy,
        )

        if field[neighbour] == 0:
            neighbour_codes.append(FREE)
        else:
            neighbour_codes.append(BLOCKED)

    return tuple(neighbour_codes)


def state_to_features(game_state: dict):
    #Wandelt den vollständigen Spielzustand in einen diskreten Zustand um

    if game_state is None:
        return None

    field = game_state["field"]
    position = tuple(game_state["self"][3])
    coins = game_state["coins"]

    coin_direction = direction_to_nearest_coin(
        field=field,
        start=position,
        coins=coins,
    )

    neighbours = local_obstacles(
        field=field,
        position=position,
    )

    return (coin_direction, *neighbours)


def distance_to_nearest_coin(field, start, coins):
    #Berechnet die kürzeste begehbare Distanz zur nächsten Münze.

    targets = set(coins)
    start = tuple(start)

    if not targets:
        return None

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        position, distance = queue.popleft()

        if position in targets:
            return distance

        for dx, dy in DIRECTIONS:
            neighbour = (
                position[0] + dx,
                position[1] + dy,
            )

            if neighbour in visited:
                continue

            if field[neighbour] != 0:
                continue

            visited.add(neighbour)
            queue.append((neighbour, distance + 1))

    return None