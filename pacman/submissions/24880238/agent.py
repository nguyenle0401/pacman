"""Ghost: simultaneous maximin with alpha-beta and cached search bounds."""
import sys
import time
from collections import Counter, deque
from pathlib import Path

# Add src to path to import the interface
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class SearchTimeout(Exception):
    pass


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught

    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo

    Current implementation: BFS evaluation and minimax with alpha-beta.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Ghost receives no settings from the loader. These match CLI defaults.
        self.pacman_speed = max(1, int(kwargs.get('pacman_speed', 2)))
        self.capture_distance = max(1, int(kwargs.get('capture_distance', 2)))
        self.max_steps = int(kwargs.get('max_steps', 200))
        self.time_budget = float(kwargs.get('time_budget', 0.15))
        self.max_depth = int(kwargs.get('max_depth', 8))
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.visits = Counter()
        self.topology = None

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _prepare_map(self, grid):
        topology = (grid.shape, (grid == 1).tobytes())
        if topology == self.topology:
            return
        self.topology = topology
        h, w = grid.shape
        # Walls are always visible: -1 is an empty tile outside sight.
        cells = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != 1}
        self.actions, self.targets, self.distances = {}, {}, {}
        for pos in cells:
            actions, targets = [(Move.STAY, pos)], {pos}
            for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                dr, dc = move.value
                neighbor = (pos[0] + dr, pos[1] + dc)
                if neighbor in cells:
                    actions.append((move, neighbor))
                current = pos
                for _ in range(self.pacman_speed):
                    nxt = (current[0] + dr, current[1] + dc)
                    if nxt not in cells:
                        break
                    targets.add(nxt)
                    current = nxt
            self.actions[pos] = actions
            self.targets[pos] = tuple(sorted(targets))

    def _distance_map(self, start):
        if start not in self.distances:
            distances, queue = {start: 0}, deque([start])
            while queue:
                pos = queue.popleft()
                for _, nxt in self.actions[pos]:
                    if nxt not in distances:
                        distances[nxt] = distances[pos] + 1
                        queue.append(nxt)
            self.distances[start] = distances
        return self.distances[start]

    def _evaluate(self, ghost, pacman):
        distance = self._distance_map(pacman).get(ghost, 1000)
        exits = len(self.actions[ghost]) - 1
        return (12 * min(distance, 40) + 4 * exits
                - (24 if exits <= 1 else 0)
                + 0.5 * min(self._manhattan(ghost, pacman), 20))

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def _search(self, ghost, pacman, depth, remaining, alpha, beta):
        self._check_time()
        if remaining <= 0:
            return 100000
        if depth == 0:
            return self._evaluate(ghost, pacman)
        key = (ghost, pacman, depth, remaining)
        original_alpha, original_beta = alpha, beta
        cached = self.table.get(key)
        if cached is not None:
            value, flag = cached
            if flag == 'exact':
                return value
            if flag == 'lower':
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value
        value = -float('inf')
        actions = sorted(self.actions[ghost],
                         key=lambda a: self._evaluate(a[1], pacman), reverse=True)
        for _, nxt in actions:
            value = max(value, self._response(nxt, pacman, depth, remaining, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        flag = ('upper' if value <= original_alpha else
                'lower' if value >= original_beta else 'exact')
        self.table[key] = (value, flag)
        return value

    def _response(self, next_ghost, pacman, depth, remaining, alpha, beta):
        # Simultaneous moves from old positions, capture only at endpoints.
        worst = float('inf')
        targets = sorted(self.targets[pacman],
                         key=lambda p: self._manhattan(next_ghost, p))
        for next_pacman in targets:
            self._check_time()
            if self._manhattan(next_ghost, next_pacman) < self.capture_distance:
                score = -100000 - remaining  # Delay unavoidable capture.
            else:
                score = self._search(next_ghost, next_pacman, depth - 1,
                                     remaining - 1, alpha, min(beta, worst))
            worst = min(worst, score)
            if worst <= alpha:
                break
        return worst

    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        self._prepare_map(map_state)
        self.visits[my_position] += 1
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        # Use current sighting, fallback to last known, or explore.
        threat = self.last_known_enemy_pos
        actions = self.actions[my_position]
        if threat is None:
            # No information about enemy - explore less visited positions.
            return max(actions, key=lambda a: (
                -self.visits[a[1]], len(self.actions[a[1]]), a[0] != Move.STAY))[0]

        # Under fog, the last sighting is only an estimate, not a guarantee.
        def fallback(action):
            move, pos = action
            safety = min(self._manhattan(pos, p) for p in self.targets[threat])
            return (safety >= self.capture_distance, self._evaluate(pos, threat),
                    -self.visits[pos], move != Move.STAY)

        actions = sorted(actions, key=fallback, reverse=True)
        best_move = actions[0][0]
        self.deadline = time.perf_counter() + self.time_budget
        self.table = {}
        remaining = max(1, self.max_steps - step_number + 1)
        # Iterative deepening preserves the last fully completed decision.
        for depth in range(1, min(self.max_depth, remaining) + 1):
            try:
                best_score, candidate = -float('inf'), best_move
                ordered = sorted(actions, key=lambda a: a[0] != best_move)
                for move, pos in ordered:
                    score = self._response(pos, threat, depth, remaining,
                                           best_score, float('inf'))
                    if score > best_score:
                        best_score, candidate = score, move
                best_move = candidate
            except SearchTimeout:
                break
        return best_move
