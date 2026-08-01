"""Merkmalsextraktion — Stufe 1 der Task-Leiter (`coin-heaven`, Navigation).

Der Zustand ist ein *Tupel*, kein Vektor: Modell A führt die Q-Tabelle als `dict`,
also muss der Zustand hashbar sein (KONZEPT §3).

    (coin_dir, n_up, n_right, n_down, n_left)   ->  5 · 3^4 = 405 Zustände

Die BFS-Helfer hier sind bewusst allgemeiner gehalten als Stufe 1 braucht
(`danger_map` wertet die volle Explosionsgeometrie aus, obwohl in `coin-heaven`
nie eine Bombe liegt). Stufe 2 braucht dieselben Funktionen für `escape_dir` und
`escape_after_bomb` — und laut KONZEPT vergiftet ein Fehler in der Fluchtlogik
alles Weitere, also gehört sie in einen eigenen, testbaren Baustein.
"""

from collections import deque

import numpy as np

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Reihenfolge identisch zu ACTIONS[:4], damit ein Richtungs-Index direkt ein
# Aktions-Index ist. environment.py:130ff: UP ist y-1, DOWN ist y+1 (Bildkoordinaten).
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Codes für ein Nachbarfeld
FREE, BLOCKED, DEADLY = 0, 1, 2

# Code für "kein Ziel erreichbar"; 1..4 sind Index in DIRECTIONS plus eins.
DIR_NONE = 0


def danger_map(field, bombs, explosion_map):
    """Schritte, bis ein Feld tödlich wird; `np.inf` wo sicher.

    Explosionsgeometrie wie `items.Bomb.get_blast_coords`: Reichweite
    `BOMB_POWER` in vier Richtungen, geht nicht um Ecken, wird **nur von
    Steinwänden** geblockt — Kisten halten die Explosion *nicht* auf.
    """
    danger = np.full(field.shape, np.inf)
    danger[explosion_map > 0] = 0

    for (bomb_x, bomb_y), timer in bombs:
        tiles = [(bomb_x, bomb_y)]
        for dx, dy in DIRECTIONS:
            for step in range(1, s.BOMB_POWER + 1):
                x, y = bomb_x + dx * step, bomb_y + dy * step
                if field[x, y] == -1:
                    break
                tiles.append((x, y))
        for x, y in tiles:
            danger[x, y] = min(danger[x, y], timer)

    return danger


def direction_to_nearest(field, start, targets, blocked=frozenset()):
    """Erster Schritt eines kürzesten Weges von `start` zum nächsten Ziel.

    Breitensuche über freie Felder. Rückgabe ist ein Richtungscode
    (`DIR_NONE` oder DIRECTIONS-Index + 1), **keine Aktion** — das Modell muss
    selbst lernen, dass es dorthin laufen sollte. Ein Merkmal, das direkt die
    beste Aktion zurückgibt, wäre laut Aufgabenstellung unzulässig.

    Die Suche ist deterministisch (Gleichstand wird über die Reihenfolge in
    DIRECTIONS gebrochen), anders als beim `rule_based_agent`, der mischt: eine
    Q-Tabelle lernt schneller, wenn derselbe Zustand immer dasselbe Merkmal
    liefert.
    """
    targets = set(targets)
    if not targets:
        return DIR_NONE

    start = (int(start[0]), int(start[1]))
    if start in targets:
        return DIR_NONE

    queue = deque()
    seen = {start}
    for index, (dx, dy) in enumerate(DIRECTIONS):
        tile = (start[0] + dx, start[1] + dy)
        if field[tile] != 0 or tile in blocked:
            continue
        seen.add(tile)
        queue.append((tile, index + 1))

    while queue:
        tile, first_step = queue.popleft()
        if tile in targets:
            return first_step
        for dx, dy in DIRECTIONS:
            neighbour = (tile[0] + dx, tile[1] + dy)
            if neighbour in seen or field[neighbour] != 0 or neighbour in blocked:
                continue
            seen.add(neighbour)
            queue.append((neighbour, first_step))

    return DIR_NONE


def neighbour_codes(field, position, danger, blocked=frozenset()):
    """Die vier angrenzenden Felder als `{FREE, BLOCKED, DEADLY}`.

    `DEADLY` heißt "explodiert jetzt gerade" (`danger <= 0`) — das reine
    Betretungsverbot. Die feineren Gefahrenstufen für das *eigene* Feld sind
    Stufe 2 (`danger`-Merkmal, 4 Werte) und gehören nicht hierher.
    """
    codes = []
    for dx, dy in DIRECTIONS:
        tile = (position[0] + dx, position[1] + dy)
        if field[tile] != 0 or tile in blocked:
            codes.append(BLOCKED)
        elif danger[tile] <= 0:
            codes.append(DEADLY)
        else:
            codes.append(FREE)
    return codes


def state_to_features(game_state: dict):
    """`game_state` -> hashbares Zustandstupel, oder `None` vor/nach der Runde."""
    if game_state is None:
        return None

    field = game_state['field']
    position = game_state['self'][3]

    # Bomben und Gegner blockieren Felder genau wie Wände (environment.tile_is_free).
    blocked = {other[3] for other in game_state['others']}
    blocked |= {tile for tile, _ in game_state['bombs']}

    danger = danger_map(field, game_state['bombs'], game_state['explosion_map'])
    coin_dir = direction_to_nearest(field, position, game_state['coins'], blocked)

    return (coin_dir, *neighbour_codes(field, position, danger, blocked))
