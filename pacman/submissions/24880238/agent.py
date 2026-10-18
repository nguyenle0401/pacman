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
        self.possible_enemies = set()
        self.previous_step = None
        self.deadline = float('inf')

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _prepare_map(self, grid):
        topology = (grid.shape, (grid == 1).tobytes(), self.pacman_speed,
                    self.capture_distance)
        if topology == self.topology:
            return
        h, w = grid.shape
        # Walls are always visible: -1 is an empty tile outside sight.
        cells = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != 1}
        all_actions, all_targets, capture_zones = {}, {}, {}
        for pos in cells:
            self._check_time()
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
            all_actions[pos] = actions
            all_targets[pos] = tuple(sorted(targets))
            radius = min(self.capture_distance - 1, h + w)
            capture_zones[pos] = tuple(
                (r, c)
                for r in range(max(0, pos[0] - radius),
                               min(h, pos[0] + radius + 1))
                for c in range(max(0, pos[1] - radius),
                               min(w, pos[1] + radius + 1))
                if (r, c) in cells
                and self._manhattan(pos, (r, c)) < self.capture_distance)
        degree = {pos: len(all_actions[pos]) - 1 for pos in cells}
        queue = deque(pos for pos in cells if degree[pos] <= 1)
        while queue:
            self._check_time()
            pos = queue.popleft()
            for _, nxt in all_actions[pos]:
                if nxt != pos and degree[nxt] > 1:
                    degree[nxt] -= 1
                    if degree[nxt] == 1:
                        queue.append(nxt)
        trapped = {pos for pos in cells if degree[pos] <= 1}
        self.topology = topology
        self.actions, self.targets = all_actions, all_targets
        self.capture_zones, self.trapped = capture_zones, trapped
        self.distances, self.pacman_distances = {}, {}
        self.possible_enemies.clear()
        self.previous_step = None
        self.last_known_enemy_pos = None
        self.visits.clear()

    def _pacman_distance_map(self, start):
        if start not in self.pacman_distances:
            distances, queue = {start: 0}, deque([start])
            while queue:
                self._check_time()
                pos = queue.popleft()
                for nxt in self.targets[pos]:
                    if nxt not in distances:
                        distances[nxt] = distances[pos] + 1
                        queue.append(nxt)
            self.pacman_distances[start] = distances
        return self.pacman_distances[start]

    def _update_belief(self, grid, enemy, step_number):
        if enemy is not None:
            enemies = {enemy}
        else:
            elapsed = (step_number - self.previous_step
                       if self.previous_step is not None else 0)
            enemies = self.possible_enemies
            if not enemies or elapsed < 0:
                enemies = set(self.actions)
            for _ in range(min(max(0, elapsed), len(self.actions))):
                self._check_time()
                expanded = self._expand_enemies(enemies)
                if expanded == enemies:
                    break
                enemies = expanded
            enemies = {pos for pos in enemies if grid[pos] == -1}
            if not enemies:
                enemies = {pos for pos in self.actions if grid[pos] == -1}
        self._check_time()
        self.possible_enemies = enemies
        self.previous_step = step_number

    def _distance_map(self, start):
        if start not in self.distances:
            distances, queue = {start: 0}, deque([start])
            while queue:
                self._check_time()
                pos = queue.popleft()
                for _, nxt in self.actions[pos]:
                    if nxt not in distances:
                        distances[nxt] = distances[pos] + 1
                        queue.append(nxt)
            self.distances[start] = distances
        return self.distances[start]

    def _expand_enemies(self, enemies):
        expanded = set()
        for pos in enemies:
            self._check_time()
            expanded.update(self.targets[pos])
        return expanded

    def _nearest_distances(self, enemies, pacman=False):
        distances = dict.fromkeys(enemies, 0)
        queue = deque(enemies)
        while queue:
            self._check_time()
            pos = queue.popleft()
            neighbors = (self.targets[pos] if pacman else
                         (nxt for _, nxt in self.actions[pos]))
            for nxt in neighbors:
                if nxt not in distances:
                    distances[nxt] = distances[pos] + 1
                    queue.append(nxt)
        return distances

    def _hidden_evaluate(self, ghost, distances, turns):
        exits = len(self.actions[ghost]) - 1
        return (12 * min(distances.get(ghost, 1000), 40)
                + 10 * min(turns.get(ghost, 1000), 20) + 4 * exits
                - (24 if exits <= 1 else 0)
                - (32 if ghost in self.trapped else 0))

    def _hidden_value(self, ghost, level, depth, remaining,
                      hazards, distances, turns, table):
        self._check_time()
        key = (ghost, level)
        if key not in table:
            if ghost in hazards[level]:
                value = -100000 - remaining + level
            elif level + 1 >= remaining:
                value = 100000
            elif level + 1 == depth:
                value = self._hidden_evaluate(ghost, distances, turns)
            else:
                value = max(self._hidden_value(
                    nxt, level + 1, depth, remaining,
                    hazards, distances, turns, table)
                    for _, nxt in self.actions[ghost])
            table[key] = value
        return table[key]

    def _hidden_move(self, actions, remaining, best_move):
        enemies = self.possible_enemies
        hazards = []
        try:
            for depth in range(1, min(self.max_depth, remaining) + 1):
                enemies = self._expand_enemies(enemies)
                danger = Counter()
                for pos in enemies:
                    self._check_time()
                    danger.update(self.capture_zones[pos])
                hazards.append(danger)
                distances = self._nearest_distances(enemies)
                turns = self._nearest_distances(enemies, pacman=True)
                if depth == 1:
                    actions = sorted(actions, key=lambda a: (
                        -danger[a[1]],
                        self._hidden_evaluate(a[1], distances, turns),
                        -self.visits[a[1]], a[0] != Move.STAY), reverse=True)
                    best_move = actions[0][0]
                table = {}
                best_score, candidate = -float('inf'), best_move
                ordered = sorted(actions, key=lambda a: a[0] != best_move)
                for move, pos in ordered:
                    score = self._hidden_value(
                        pos, 0, depth, remaining, hazards, distances, turns, table)
                    if score > best_score:
                        best_score, candidate = score, move
                best_move = candidate
                if best_score < -100000 or best_score == 100000:
                    break
        except SearchTimeout:
            pass
        return best_move

    def _quick_move(self, grid, ghost, enemy):
        h, w = grid.shape
        best_move, best_score = Move.STAY, None
        for move in Move:
            dr, dc = move.value
            pos = (ghost[0] + dr, ghost[1] + dc)
            if not (0 <= pos[0] < h and 0 <= pos[1] < w) or grid[pos] == 1:
                continue
            if enemy is None:
                score = (-self.visits[pos], move == Move.STAY)
            else:
                targets = [enemy]
                for direction in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                    er, ec = enemy
                    vr, vc = direction.value
                    for _ in range(min(self.pacman_speed, max(h, w))):
                        er, ec = er + vr, ec + vc
                        if not (0 <= er < h and 0 <= ec < w) or grid[er, ec] == 1:
                            break
                        targets.append((er, ec))
                safety = min(self._manhattan(pos, target) for target in targets)
                score = (safety >= self.capture_distance, safety,
                         self._manhattan(pos, enemy), -self.visits[pos])
            if best_score is None or score > best_score:
                best_score, best_move = score, move
        return best_move

    def _evaluate(self, ghost, pacman):
        distance = self._distance_map(pacman).get(ghost, 1000)
        exits = len(self.actions[ghost]) - 1
        turns = self._pacman_distance_map(pacman).get(ghost, 1000)
        return (12 * min(distance, 40) + 10 * min(turns, 20) + 4 * exits
                - (24 if exits <= 1 else 0)
                - (32 if ghost in self.trapped else 0)
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
        self.deadline = time.perf_counter() + self.time_budget
        if self.previous_step is not None and step_number <= self.previous_step:
            self.possible_enemies.clear()
            self.previous_step = None
            self.last_known_enemy_pos = None
            self.visits.clear()
        best_move = self._quick_move(map_state, my_position, enemy_position)
        try:
            self._check_time()
            self._prepare_map(map_state)
            self._update_belief(map_state, enemy_position, step_number)
            self.visits[my_position] += 1
            # Update memory if enemy is visible
            if enemy_position is not None:
                self.last_known_enemy_pos = enemy_position
            # Use current sighting, fallback to last known, or explore.
            threat = self.last_known_enemy_pos
            actions = self.actions[my_position]
            remaining = max(1, self.max_steps - step_number + 1)
            if enemy_position is None and self.possible_enemies:
                return self._hidden_move(actions, remaining, best_move)
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
            self.table = {}
            # Iterative deepening preserves the last fully completed decision.
            for depth in range(1, min(self.max_depth, remaining) + 1):
                best_score, candidate = -float('inf'), best_move
                ordered = sorted(actions, key=lambda a: a[0] != best_move)
                for move, pos in ordered:
                    score = self._response(pos, threat, depth, remaining,
                                           best_score, float('inf'))
                    if score > best_score:
                        best_score, candidate = score, move
                best_move = candidate
        except SearchTimeout:
            pass
        return best_move
