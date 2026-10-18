"""Ghost: simultaneous maximin with alpha-beta and cached search bounds."""
import sys
import time
from collections import Counter, deque
from pathlib import Path
import numpy as np

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
        self.strategy = None
        self.belief = None
        self.last_sighting = None
        self.current_position = None
        self.model_prior = np.asarray([0.4, 0.25, 0.25, 0.1])
        self.model_weights = self.model_prior.copy()
        self.model_observations = 0
        self.fog_enabled = False
        self.vision_radius = 0
        self.visibility = {}
        self.decision_mode = 'fallback'
        self.bfs_strategy = None
        self.ordered_targets = {}
        self.bfs_consistent = True
        self.bfs_observations = 0

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
        escape_routes = self._find_escape_routes(all_actions)
        self.topology = topology
        self.actions, self.targets = all_actions, all_targets
        self.capture_zones, self.trapped = capture_zones, trapped
        self.escape_routes = escape_routes
        self.distances, self.pacman_distances = {}, {}
        self.strategy = None
        self.belief = None
        self.last_sighting = None
        self.model_weights = self.model_prior.copy()
        self.model_observations = 0
        self.fog_enabled = False
        self.vision_radius = 0
        self.visibility = {}
        self.bfs_strategy = None
        self.ordered_targets = {}
        self.bfs_consistent = True
        self.bfs_observations = 0
        self.possible_enemies.clear()
        self.previous_step = None
        self.last_known_enemy_pos = None
        self.visits.clear()

    def _find_escape_routes(self, actions):
        neighbors = {pos: tuple(nxt for _, nxt in moves if nxt != pos)
                     for pos, moves in actions.items()}
        discovered, low, parent, children = {}, {}, {}, {}
        gates = set()
        for root in neighbors:
            if root in discovered:
                continue
            discovered[root] = low[root] = len(discovered)
            parent[root], children[root] = None, 0
            stack = [(root, iter(neighbors[root]))]
            while stack:
                self._check_time()
                pos, pending = stack[-1]
                nxt = next(pending, None)
                if nxt is None:
                    stack.pop()
                    previous = parent[pos]
                    if previous is None:
                        if children[pos] > 1:
                            gates.add(pos)
                    else:
                        low[previous] = min(low[previous], low[pos])
                        if (parent[previous] is not None
                                and low[pos] >= discovered[previous]):
                            gates.add(previous)
                elif nxt not in discovered:
                    discovered[nxt] = low[nxt] = len(discovered)
                    parent[nxt], children[nxt] = pos, 0
                    children[pos] += 1
                    stack.append((nxt, iter(neighbors[nxt])))
                elif nxt != parent[pos]:
                    low[pos] = min(low[pos], discovered[nxt])

        routes = {pos: [] for pos in neighbors}
        for gate in sorted(gates):
            seen, regions = {gate}, []
            for start in neighbors[gate]:
                if start in seen:
                    continue
                region, queue = {start}, deque([start])
                seen.add(start)
                while queue:
                    self._check_time()
                    pos = queue.popleft()
                    for nxt in neighbors[pos]:
                        if nxt not in seen:
                            seen.add(nxt)
                            region.add(nxt)
                            queue.append(nxt)
                regions.append(region)
            size = sum(map(len, regions))
            for region in regions:
                if 2 * len(region) > size:
                    continue
                distances = {pos: 1 for pos in neighbors[gate] if pos in region}
                queue = deque(distances)
                while queue:
                    self._check_time()
                    pos = queue.popleft()
                    routes[pos].append((gate, distances[pos], len(region)))
                    for nxt in neighbors[pos]:
                        if nxt in region and nxt not in distances:
                            distances[nxt] = distances[pos] + 1
                            queue.append(nxt)
        return {pos: tuple(values) for pos, values in routes.items()}

    def _escape_penalty(self, ghost, turns):
        penalty = 0
        for gate, steps, size in self.escape_routes[ghost]:
            arrival = min(turns.get(pos, 1000) for pos in self.capture_zones[gate])
            risk = max(0, min(12, 3 + steps - arrival))
            penalty = max(penalty, 12 * risk * min(1, 16 / size))
        return penalty

    def _prepare_strategy(self):
        if self.strategy is not None or len(self.actions) > 512:
            return
        cells = sorted(self.actions)
        index = {pos: i for i, pos in enumerate(cells)}
        ghost_width = max(map(len, self.actions.values()))
        pacman_width = max(map(len, self.targets.values()))
        ghost_moves, pacman_moves = [], []
        for pos in cells:
            self._check_time()
            ghost = [index[nxt] for _, nxt in self.actions[pos]]
            pacman = [index[nxt] for nxt in self.targets[pos]]
            ghost_moves.append(ghost + [index[pos]] * (ghost_width - len(ghost)))
            pacman_moves.append(pacman + [index[pos]] * (pacman_width - len(pacman)))
        coordinates = np.asarray(cells)
        safe = (np.abs(coordinates[:, None] - coordinates[None, :]).sum(axis=2)
                >= self.capture_distance)
        ranks = np.full(safe.shape, -1, dtype=np.int32)
        ranks[~safe] = 0
        self._check_time()
        self.strategy = {
            'index': index,
            'ghost_moves': np.asarray(ghost_moves, dtype=np.int32),
            'pacman_moves': np.asarray(pacman_moves, dtype=np.int32),
            'safe': safe,
            'alive': safe.copy(),
            'ranks': ranks,
            'depth': 0,
            'solved': False,
        }

    def _strategy_move(self, actions, ghost, pacman, remaining):
        self._prepare_strategy()
        strategy = self.strategy
        if strategy is None:
            return None
        index = strategy['index']
        g, p = index[ghost], index[pacman]
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * 0.65
        while (not strategy['solved'] and strategy['ranks'][g, p] < 0
               and strategy['depth'] < remaining and time.perf_counter() < stop):
            self._check_time()
            alive = strategy['alive']
            responses = alive[:, strategy['pacman_moves']].all(axis=2)
            following = responses[strategy['ghost_moves']].any(axis=1)
            following &= strategy['safe']
            lost = alive & ~following
            depth = strategy['depth'] + 1
            strategy['ranks'][lost] = depth
            strategy['alive'] = following
            strategy['depth'] = depth
            strategy['solved'] = not lost.any()
        self._check_time()
        if (not strategy['solved'] and strategy['ranks'][g, p] < 0
                and strategy['depth'] < remaining):
            return None
        targets = [index[pos] for pos in self.targets[pacman]]
        best_move, best_score = None, None
        for move, pos in actions:
            self._check_time()
            values = strategy['ranks'][index[pos], targets]
            outcomes = tuple(sorted(remaining if value < 0 else
                                    min(remaining, int(value)) for value in values))
            score = (outcomes, self._evaluate(pos, pacman),
                     -self.visits[pos], move != Move.STAY)
            if best_score is None or score > best_score:
                best_move, best_score = move, score
        self.decision_mode = 'robust_visible'
        if (0 < strategy['ranks'][g, p] <= remaining
                and self.bfs_consistent
                and (self.model_weights[0] >= self.model_weights[1:].max()
                     or self.bfs_observations >= 2)
                and (not self.fog_enabled or (self.model_observations >= 2
                                             and self.model_weights[0] >= 0.55))):
            response = self._bfs_response_move(actions, ghost, pacman, remaining)
            if response is not None:
                self.decision_mode = 'bfs_survival'
                return response
        if (strategy['ranks'][g, p] > 0
                and strategy['ranks'][g, p] <= remaining
                and self._model_exploitable()):
            self._prepare_belief()
            if self.belief is not None:
                probabilities = np.zeros(len(self.belief['nodes']))
                probabilities[self.belief['index'][pacman]] = 1
                self.belief['probabilities'] = probabilities
                safe = [move for move, pos in actions
                        if all(self._manhattan(pos, target) >= self.capture_distance
                               for target in self.targets[pacman])]
                self.decision_mode = 'exploit'
                return self._forecast_move(remaining, best_move, set(safe) if safe else None)
        return best_move

    def _ordered_pacman_targets(self, pos):
        if pos not in self.ordered_targets:
            targets = []
            for move in Move:
                current = pos
                dr, dc = move.value
                for _ in range(min(self.pacman_speed, max(self.topology[0]))):
                    nxt = (current[0] + dr, current[1] + dc)
                    if nxt in self.actions:
                        current = nxt
                    targets.append(current)
            self.ordered_targets[pos] = tuple(targets)
        return self.ordered_targets[pos]

    def _prepare_bfs_strategy(self):
        if self.bfs_strategy is not None:
            return
        self._prepare_strategy()
        if self.strategy is None:
            return
        nodes = sorted(self.actions)
        index = self.strategy['index']
        ordered, distances = [], []
        for ghost in nodes:
            self._check_time()
            ordered.append([index[pos] for pos in self._ordered_pacman_targets(ghost)])
            distance = self._distance_map(ghost)
            distances.append([distance.get(pos, 1000) for pos in nodes])
        ordered = np.asarray(ordered, dtype=np.int32)
        costs = np.asarray(distances, dtype=np.int32)[:, ordered]
        policy = ordered[np.arange(len(nodes))[None, :], costs.argmin(axis=2)]
        safe = self.strategy['safe']
        ranks = np.full(safe.shape, -1, dtype=np.int32)
        ranks[~safe] = 0
        self._check_time()
        self.bfs_strategy = {
            'index': index, 'policy': policy, 'ranks': ranks,
            'alive': safe.copy(), 'depth': 0, 'solved': False,
        }

    def _bfs_response_move(self, actions, ghost, pacman, remaining):
        self._prepare_bfs_strategy()
        response = self.bfs_strategy
        if response is None:
            return None
        index = response['index']
        g, p = index[ghost], index[pacman]
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * 0.65
        while (not response['solved'] and response['ranks'][g, p] < 0
               and response['depth'] < remaining and time.perf_counter() < stop):
            self._check_time()
            following = response['alive'][self.strategy['ghost_moves'][:, :, None],
                                           response['policy'][:, None, :]].any(axis=1)
            following &= self.strategy['safe']
            lost = response['alive'] & ~following
            depth = response['depth'] + 1
            response['ranks'][lost] = depth
            response['alive'], response['depth'] = following, depth
            response['solved'] = not lost.any()
        self._check_time()
        if (not response['solved'] and response['ranks'][g, p] < 0
                and response['depth'] < remaining):
            return None
        target = response['policy'][g, p]
        best_move, best_score = None, None
        for move, pos in actions:
            self._check_time()
            rank = int(response['ranks'][index[pos], target])
            lifetime = remaining if rank < 0 else min(remaining, rank)
            safe = all(self._manhattan(pos, enemy) >= self.capture_distance
                       for enemy in self.targets[pacman])
            score = (lifetime, safe, self._evaluate(pos, pacman),
                     -self.visits[pos], move != Move.STAY)
            if best_score is None or score > best_score:
                best_move, best_score = move, score
        return best_move

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
        self._update_probabilities(enemy, step_number)

    def _prepare_belief(self):
        if self.belief is not None or len(self.actions) > 512:
            return
        nodes = sorted(self.actions)
        index = {pos: i for i, pos in enumerate(nodes)}
        width = max(map(len, self.targets.values()))
        edges = []
        for pos in nodes:
            self._check_time()
            targets = [index[nxt] for nxt in self.targets[pos]]
            edges.append(targets + [index[pos]] * (width - len(targets)))
        coordinates = np.asarray(nodes)
        capture = (np.abs(coordinates[:, None] - coordinates[None, :]).sum(axis=2)
                   < self.capture_distance)
        self._check_time()
        self.belief = {
            'nodes': nodes,
            'index': index,
            'edges': np.asarray(edges, dtype=np.int32),
            'capture': capture,
            'policies': {},
            'models': {},
            'geometry': {},
            'probabilities': None,
            'observer': None,
            'step': None,
        }

    def _update_probabilities(self, enemy, step_number):
        self._prepare_belief()
        belief = self.belief
        if belief is None:
            return
        nodes, index = belief['nodes'], belief['index']
        learned = None
        bfs_consistent = self.bfs_consistent
        bfs_observations = self.bfs_observations
        if enemy is not None:
            if self.last_sighting is not None:
                previous, observer, previous_step = self.last_sighting
                if previous_step + 1 == step_number:
                    models = self._pursuit_models(observer)
                    predicted = min(self._ordered_pacman_targets(previous),
                                    key=lambda pos: self._distance_map(observer).get(pos, 1000))
                    bfs_consistent = bfs_consistent and predicted == enemy
                    bfs_observations = bfs_observations + 1 if predicted == enemy else 0
                    outcomes = belief['edges'][index[previous]] == index[enemy]
                    likelihood = models[:, index[previous], outcomes].sum(axis=1)
                    if likelihood.sum() > 0:
                        learned = self.model_weights * (likelihood + 0.02)
                        learned = 0.98 * learned / learned.sum() + 0.02 * self.model_prior
            probabilities = np.zeros(len(nodes))
            probabilities[index[enemy]] = 1
        else:
            if belief['probabilities'] is not None:
                probabilities = belief['probabilities'].copy()
                observer, previous = belief['observer'], belief['step']
            elif self.last_sighting is not None:
                position, observer, previous = self.last_sighting
                probabilities = np.zeros(len(nodes))
                probabilities[index[position]] = 1
            else:
                probabilities = np.full(len(nodes), 1 / len(nodes))
                observer, previous = None, step_number
            if observer is not None:
                for _ in range(min(max(0, step_number - previous), len(nodes))):
                    self._check_time()
                    probabilities = self._forecast(probabilities, observer)
            mask = np.asarray([pos in self.possible_enemies for pos in nodes])
            probabilities *= mask
            if probabilities.sum() <= 1e-12:
                probabilities = mask.astype(float)
            probabilities += mask * 1e-6
            mass = probabilities.sum()
            if mass <= 0:
                return
            probabilities /= mass
        self._check_time()
        self.bfs_consistent = bfs_consistent
        self.bfs_observations = bfs_observations
        if learned is not None:
            self.model_weights = learned
            self.model_observations += 1
            belief['policies'].clear()
        belief['probabilities'] = probabilities
        belief['observer'] = self.current_position
        belief['step'] = step_number
        if enemy is not None:
            self.last_sighting = (enemy, self.current_position, step_number)

    def _pursuit_policy(self, ghost):
        belief = self.belief
        if ghost in belief['policies']:
            return belief['policies'][ghost]
        models = self._pursuit_models(ghost)
        weights = np.tensordot(self.model_weights, models, axes=(0, 0))
        self._check_time()
        belief['policies'][ghost] = weights
        return weights

    def _pursuit_models(self, ghost):
        belief = self.belief
        if ghost in belief['models']:
            return belief['models'][ghost]
        models = np.zeros((4,) + belief['edges'].shape)
        distances = self._distance_map(ghost)
        for row, pacman in enumerate(belief['nodes']):
            self._check_time()
            targets = self.targets[pacman]
            costs = [distances.get(pos, 1000) for pos in targets]
            closest = min(costs)
            count = costs.count(closest)
            for column, cost in enumerate(costs):
                models[0, row, column] = (cost == closest) / count
                models[3, row, column] = 1 / len(targets)
            dr, dc = ghost[0] - pacman[0], ghost[1] - pacman[1]
            vertical = Move.DOWN if dr > 0 else Move.UP if dr < 0 else None
            horizontal = Move.RIGHT if dc > 0 else Move.LEFT if dc < 0 else None
            for model, preferred in enumerate(((vertical, horizontal), (horizontal, vertical)), 1):
                selected = None
                for direction in preferred:
                    if direction is None:
                        continue
                    steps = min(self.pacman_speed, abs(dr) if direction in
                                (Move.UP, Move.DOWN) else abs(dc))
                    pos = pacman
                    for _ in range(steps):
                        rr, cc = direction.value
                        nxt = (pos[0] + rr, pos[1] + cc)
                        if nxt not in self.actions:
                            break
                        pos = nxt
                    if pos != pacman:
                        selected = pos
                        break
                if selected is not None:
                    models[model, row, targets.index(selected)] = 1
                else:
                    endpoints = []
                    for direction in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                        pos = pacman
                        for _ in range(self.pacman_speed):
                            rr, cc = direction.value
                            nxt = (pos[0] + rr, pos[1] + cc)
                            if nxt not in self.actions:
                                break
                            pos = nxt
                        if pos != pacman:
                            endpoints.append(pos)
                    if not endpoints:
                        endpoints = [pacman]
                    for pos in endpoints:
                        models[model, row, targets.index(pos)] += 1 / len(endpoints)
        self._check_time()
        belief['models'][ghost] = models
        return models

    def _model_exploitable(self):
        return self.model_observations >= 2 and self.model_weights[1:3].max() >= 0.55

    def _visible_from(self, pos):
        key = (pos, self.vision_radius)
        if key not in self.visibility:
            cells = {pos}
            for direction in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                dr, dc = direction.value
                r, c = pos
                for _ in range(self.vision_radius):
                    r, c = r + dr, c + dc
                    if (r, c) not in self.actions:
                        break
                    cells.add((r, c))
            self.visibility[key] = tuple(cells)
        return self.visibility[key]

    def _information_gain(self, pos, probabilities):
        mass = probabilities.sum()
        if not self.fog_enabled or mass <= 0:
            return 0
        index = self.belief['index']
        visible = probabilities[[index[cell] for cell in self._visible_from(pos)]] / mass
        outside = max(0, 1 - visible.sum())
        outcomes = np.append(visible[visible > 0], outside)
        outcomes = outcomes[outcomes > 0]
        return float(-(outcomes * np.log2(outcomes)).sum())

    def _turn_geometry(self, pos):
        geometry = self.belief['geometry']
        if pos not in geometry:
            distances = self._distance_map(pos)
            turns = self._pacman_distance_map(pos)
            space, delay = [], []
            for enemy in self.belief['nodes']:
                distance = distances.get(enemy, 1000)
                space.append(min(40, distance))
                ideal = (distance + self.pacman_speed - 1) // self.pacman_speed
                delay.append(min(4, max(0, turns.get(enemy, 1000) - ideal))
                             if distance < 1000 else 0)
            directions = {move for move, _ in self.actions[pos] if move != Move.STAY}
            corner = (self.pacman_speed > 1
                      and bool(directions & {Move.UP, Move.DOWN})
                      and bool(directions & {Move.LEFT, Move.RIGHT}))
            self._check_time()
            geometry[pos] = (np.asarray(space), np.asarray(delay), int(corner))
        return geometry[pos]

    def _forecast(self, probabilities, ghost):
        self._check_time()
        belief = self.belief
        return np.bincount(
            belief['edges'].ravel(),
            weights=(probabilities[:, None] * self._pursuit_policy(ghost)).ravel(),
            minlength=len(belief['nodes']))

    def _forecast_move(self, remaining, best_move, allowed=None):
        belief = self.belief
        states = [(self.current_position, belief['probabilities'], None)]
        try:
            for depth in range(1, min(4, self.max_depth, remaining) + 1):
                candidates = []
                for ghost, probabilities, first in states:
                    prediction = self._forecast(probabilities, ghost)
                    for move, pos in self.actions[ghost]:
                        if depth == 1 and allowed is not None and move not in allowed:
                            continue
                        self._check_time()
                        following = prediction.copy()
                        following[belief['capture'][belief['index'][pos]]] = 0
                        mass = following.sum()
                        distances, turns, corner = self._turn_geometry(pos)
                        space = float(following @ distances)
                        information = self._information_gain(pos, following)
                        turn_delay = float(following @ turns)
                        utility = (10000 * mass + 0.4 * space
                                   + 0.25 * mass * information + 0.3 * turn_delay
                                   + 0.2 * mass * corner
                                   + mass * (len(self.actions[pos]) - 1
                                             - 4 * (pos in self.trapped)
                                             - 0.1 * self.visits[pos]))
                        candidates.append((utility, first if first is not None else move,
                                           pos, following))
                candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                unique, seen = [], set()
                for candidate in candidates:
                    key = (candidate[1], candidate[2])
                    if key not in seen:
                        unique.append(candidate)
                        seen.add(key)
                best_move = unique[0][1]
                states = [(pos, probabilities, first)
                          for _, first, pos, probabilities in unique[:8]]
        except SearchTimeout:
            pass
        return best_move

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
                - (32 if ghost in self.trapped else 0)
                - self._escape_penalty(ghost, turns))

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
        self.decision_mode = 'robust_hidden'
        if self.belief is not None and self.belief['probabilities'] is not None:
            distances = self._distance_map(self.current_position)
            nearest = min((distances.get(pos, 1000) for pos in self.possible_enemies),
                          default=1000)
            if nearest <= self.pacman_speed + self.capture_distance - 1:
                self.decision_mode = 'forecast'
                return self._forecast_move(remaining, best_move)
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
        pacman_distances = self._pacman_distance_map(pacman)
        turns = pacman_distances.get(ghost, 1000)
        return (12 * min(distance, 40) + 10 * min(turns, 20) + 4 * exits
                - (24 if exits <= 1 else 0)
                - (32 if ghost in self.trapped else 0)
                - self._escape_penalty(ghost, pacman_distances)
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
        self.current_position = my_position
        self.decision_mode = 'fallback'
        if self.previous_step is not None and step_number <= self.previous_step:
            self.possible_enemies.clear()
            self.previous_step = None
            self.last_known_enemy_pos = None
            self.visits.clear()
            self.belief = None
            self.last_sighting = None
            self.model_weights = self.model_prior.copy()
            self.model_observations = 0
            self.vision_radius = 0
            self.visibility = {}
            self.bfs_consistent = True
            self.bfs_observations = 0
        best_move = self._quick_move(map_state, my_position, enemy_position)
        try:
            self._check_time()
            self._prepare_map(map_state)
            self.fog_enabled = bool((map_state == -1).any())
            if self.fog_enabled:
                visible = np.argwhere(map_state == 0)
                radius = max((abs(int(r) - my_position[0]) + abs(int(c) - my_position[1])
                              for r, c in visible), default=0)
                self.vision_radius = max(self.vision_radius, radius)
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
            if enemy_position is not None:
                strategy_move = self._strategy_move(
                    actions, my_position, enemy_position, remaining)
                if strategy_move is not None:
                    return strategy_move
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
