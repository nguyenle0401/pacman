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


MAX_TABLE_CELLS = 512
UNREACHABLE_DISTANCE = 1000
TERMINAL_SCORE = 100000
STRATEGY_TIME_FRACTION = 0.65
MODEL_BFS = 0
MODEL_GREEDY_VERTICAL = 1
MODEL_GREEDY_HORIZONTAL = 2
MODEL_RANDOM = 3
MIN_MODEL_OBSERVATIONS = 2
MODEL_CONFIDENCE_THRESHOLD = 0.55
FORECAST_HORIZON = 4
FORECAST_BEAM_WIDTH = 8


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
        self.deadline = float('inf')
        self.maximin_strategy = None
        self.current_position = None
        self.model_prior = np.asarray([0.4, 0.25, 0.25, 0.1])
        self.fog_enabled = False
        self.decision_mode = 'fallback'
        self.bfs_response_strategy = None
        self.ordered_target_cache = {}
        self._reset_episode_state()

    def _reset_episode_state(self):
        # Xóa ký ức và suy đoán của trận cũ; giữ các bảng chiến lược tĩnh theo bản đồ.
        self.possible_enemies.clear()
        self.previous_step = None
        self.last_known_enemy_pos = None
        self.visits.clear()
        self.belief = None
        self.last_sighting = None
        self.model_weights = self.model_prior.copy()
        self.model_observations = 0
        self.vision_radius = 0
        self.visibility_cache = {}
        self.bfs_consistent = True
        self.bfs_observations = 0

    @staticmethod
    def _manhattan(a, b):
        # Khoảng cách hàng + cột giữa hai ô; không xét tường hay đường vòng.
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _prepare_map(self, grid):
        # Dựng nước đi, vùng bắt và lối thoát từ grid; chỉ dựng lại khi bản đồ/luật đổi.
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
        self.ghost_actions, self.pacman_targets = all_actions, all_targets
        self.capture_zones, self.trapped = capture_zones, trapped
        self.escape_routes = escape_routes
        self.maze_distance_cache, self.pacman_turn_cache = {}, {}
        self.maximin_strategy = None
        self.fog_enabled = False
        self.bfs_response_strategy = None
        self.ordered_target_cache = {}
        self._reset_episode_state()

    def _find_escape_routes(self, actions):
        # Tìm vùng có một cửa thoát; mỗi ô lưu các bộ (cửa, số bước tới cửa, số ô trong vùng).
        neighbors = {pos: tuple(nxt for _, nxt in moves if nxt != pos)
                     for pos, moves in actions.items()}
        # Tarjan tìm các ô nút thắt: bỏ ô đó có thể chia mê cung thành nhiều vùng.
        # low lưu mốc duyệt sớm nhất có thể quay về mà không qua cạnh cha.
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
        # Phạt vị trí dễ bị chặn cửa; turns là bảng số lượt Pacman cần để tới từng ô.
        penalty = 0
        for gate, steps, size in self.escape_routes[ghost]:
            arrival = min(turns.get(pos, UNREACHABLE_DISTANCE) for pos in self.capture_zones[gate])
            risk = max(0, min(12, 3 + steps - arrival))
            penalty = max(penalty, 12 * risk * min(1, 16 / size))
        return penalty

    def _prepare_maximin_strategy(self):
        # Khởi tạo bảng sống sót cho mọi cặp vị trí và các nước hợp lệ của hai bên.
        if self.maximin_strategy is not None or len(self.ghost_actions) > MAX_TABLE_CELLS:
            return
        cells = sorted(self.ghost_actions)
        index = {pos: i for i, pos in enumerate(cells)}
        ghost_width = max(map(len, self.ghost_actions.values()))
        pacman_width = max(map(len, self.pacman_targets.values()))
        ghost_moves, pacman_moves = [], []
        for pos in cells:
            self._check_time()
            ghost = [index[nxt] for _, nxt in self.ghost_actions[pos]]
            pacman = [index[nxt] for nxt in self.pacman_targets[pos]]
            ghost_moves.append(ghost + [index[pos]] * (ghost_width - len(ghost)))
            pacman_moves.append(pacman + [index[pos]] * (pacman_width - len(pacman)))
        coordinates = np.asarray(cells)
        safe = (np.abs(coordinates[:, None] - coordinates[None, :]).sum(axis=2)
                >= self.capture_distance)
        # Bảng trạng thái có hai trục: [chỉ số ô Ghost, chỉ số ô Pacman].
        # ranks = 0: đã bị bắt; k > 0: Pacman tối ưu buộc bắt trong tối đa k lượt.
        # -1: chưa chứng minh bị bắt; khi solved=True thì có thể né vô hạn trong mô hình.
        # alive đánh dấu các trạng thái còn sống được sau số lượt depth đã xét.
        ranks = np.full(safe.shape, -1, dtype=np.int32)
        ranks[~safe] = 0
        self._check_time()
        self.maximin_strategy = {
            'index': index,
            'ghost_moves': np.asarray(ghost_moves, dtype=np.int32),
            'pacman_moves': np.asarray(pacman_moves, dtype=np.int32),
            'safe': safe,
            'alive': safe.copy(),
            'ranks': ranks,
            'depth': 0,
            'solved': False,
        }

    def _choose_maximin_move(self, actions, ghost, pacman, remaining):
        # Chọn Move tốt nhất trước phản ứng xấu nhất của Pacman.
        # Trả None nếu bảng chưa tính đủ trong thời gian cho phép.
        self._prepare_maximin_strategy()
        strategy = self.maximin_strategy
        if strategy is None:
            return None
        index = strategy['index']
        ghost_index, pacman_index = index[ghost], index[pacman]
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * STRATEGY_TIME_FRACTION
        while (not strategy['solved'] and strategy['ranks'][ghost_index, pacman_index] < 0
               and strategy['depth'] < remaining and time.perf_counter() < stop):
            self._check_time()
            survivable = strategy['alive']
            # all xét mọi nước Pacman cho từng ô Ghost có thể tới.
            # any tìm một nước Ghost sống được trước tất cả các phản ứng đó.
            safe_responses = survivable[:, strategy['pacman_moves']].all(axis=2)
            next_survivable = safe_responses[strategy['ghost_moves']].any(axis=1)
            next_survivable &= strategy['safe']
            lost = survivable & ~next_survivable
            depth = strategy['depth'] + 1
            strategy['ranks'][lost] = depth
            strategy['alive'] = next_survivable
            strategy['depth'] = depth
            strategy['solved'] = not lost.any()
        self._check_time()
        if (not strategy['solved'] and strategy['ranks'][ghost_index, pacman_index] < 0
                and strategy['depth'] < remaining):
            return None
        targets = [index[pos] for pos in self.pacman_targets[pacman]]
        best_move, best_score = None, None
        for move, pos in actions:
            self._check_time()
            values = strategy['ranks'][index[pos], targets]
            # Tuple tăng dần giúp so kết quả xấu nhất trước, rồi mới các kết quả còn lại.
            outcomes = tuple(sorted(remaining if value < 0 else
                                    min(remaining, int(value)) for value in values))
            score = (outcomes, self._evaluate(pos, pacman),
                     -self.visits[pos], move != Move.STAY)
            if best_score is None or score > best_score:
                best_move, best_score = move, score
        return best_move

    def _bfs_model_supported(self):
        # Kiểm tra giả thuyết đối thủ dùng BFS còn khớp và đủ tin cậy để áp dụng.
        bfs_weight = self.model_weights[MODEL_BFS]
        bfs_is_likely = bfs_weight >= self.model_weights[MODEL_GREEDY_VERTICAL:].max()
        repeated_matches = self.bfs_observations >= MIN_MODEL_OBSERVATIONS
        reliable_under_fog = (not self.fog_enabled
                              or (self.model_observations >= MIN_MODEL_OBSERVATIONS
                                  and bfs_weight >= MODEL_CONFIDENCE_THRESHOLD))
        return (self.bfs_consistent and (bfs_is_likely or repeated_matches)
                and reliable_under_fog)

    def _choose_visible_move(self, actions, ghost, pacman, remaining):
        # Khi thấy Pacman, ưu tiên maximin rồi xét BFS hoặc kiểu đuổi đã học.
        # Trả Move; None để step chuyển sang tìm kiếm dự phòng.
        best_move = self._choose_maximin_move(actions, ghost, pacman, remaining)
        if best_move is None:
            return None
        self.decision_mode = 'robust_visible'
        index = self.maximin_strategy['index']
        capture_turn = self.maximin_strategy['ranks'][index[ghost], index[pacman]]
        # Giữ phương án maximin nếu còn bảo đảm sống tới hết trận.
        # Chỉ thử mô hình đối thủ khi phương án bảo thủ đã chứng minh sẽ thua.
        if not 0 < capture_turn <= remaining:
            return best_move
        if self._bfs_model_supported():
            bfs_move = self._choose_bfs_survival_move(actions, ghost, pacman, remaining)
            if bfs_move is not None:
                self.decision_mode = 'bfs_survival'
                return bfs_move
        if self._model_exploitable():
            self._prepare_belief()
            if self.belief is not None:
                probabilities = np.zeros(len(self.belief['nodes']))
                probabilities[self.belief['index'][pacman]] = 1
                self.belief['probabilities'] = probabilities
                safe = [move for move, pos in actions
                        if all(self._manhattan(pos, target) >= self.capture_distance
                               for target in self.pacman_targets[pacman])]
                self.decision_mode = 'exploit'
                return self._forecast_move(remaining, best_move, set(safe) if safe else None)
        return best_move

    def _ordered_pacman_targets(self, pos):
        # Liệt kê đích theo thứ tự Move và số bước; giữ đích trùng để phá hòa ổn định.
        if pos not in self.ordered_target_cache:
            targets = []
            for move in Move:
                current = pos
                dr, dc = move.value
                for _ in range(min(self.pacman_speed, max(self.topology[0]))):
                    nxt = (current[0] + dr, current[1] + dc)
                    if nxt in self.ghost_actions:
                        current = nxt
                    targets.append(current)
            self.ordered_target_cache[pos] = tuple(targets)
        return self.ordered_target_cache[pos]

    def _prepare_bfs_strategy(self):
        # Khởi tạo bảng nước đi BFS dự đoán và bảng sống sót cho mọi cặp vị trí.
        if self.bfs_response_strategy is not None:
            return
        self._prepare_maximin_strategy()
        if self.maximin_strategy is None:
            return
        nodes = sorted(self.ghost_actions)
        index = self.maximin_strategy['index']
        ordered, distances = [], []
        for ghost in nodes:
            self._check_time()
            ordered.append([index[pos] for pos in self._ordered_pacman_targets(ghost)])
            distance = self._distance_map(ghost)
            distances.append([distance.get(pos, UNREACHABLE_DISTANCE) for pos in nodes])
        ordered = np.asarray(ordered, dtype=np.int32)
        costs = np.asarray(distances, dtype=np.int32)[:, ordered]
        # policy[g, p] là ô Pacman dự đoán sẽ tới khi nhắm vị trí Ghost cũ.
        # argmin lấy đích đầu tiên khi hòa, theo thứ tự Move của mô hình BFS này.
        policy = ordered[np.arange(len(nodes))[None, :], costs.argmin(axis=2)]
        safe = self.maximin_strategy['safe']
        ranks = np.full(safe.shape, -1, dtype=np.int32)
        ranks[~safe] = 0
        self._check_time()
        self.bfs_response_strategy = {
            'index': index, 'policy': policy, 'ranks': ranks,
            'alive': safe.copy(), 'depth': 0, 'solved': False,
        }

    def _choose_bfs_survival_move(self, actions, ghost, pacman, remaining):
        # Chọn Move kéo dài thời gian sống nhất trước chính sách BFS giả định.
        # Trả None nếu bảng chưa tính đủ để quyết định.
        self._prepare_bfs_strategy()
        response_table = self.bfs_response_strategy
        if response_table is None:
            return None
        index = response_table['index']
        ghost_index, pacman_index = index[ghost], index[pacman]
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * STRATEGY_TIME_FRACTION
        while (not response_table['solved']
               and response_table['ranks'][ghost_index, pacman_index] < 0
               and response_table['depth'] < remaining and time.perf_counter() < stop):
            self._check_time()
            # Broadcast tạo các nhánh [ô Ghost, lựa chọn Ghost, ô Pacman].
            # Pacman chỉ dùng nước dự đoán của mô hình BFS; any chọn nhánh Ghost sống được.
            ghost_moves = self.maximin_strategy['ghost_moves'][:, :, None]
            pacman_moves = response_table['policy'][:, None, :]
            next_survivable = response_table['alive'][ghost_moves, pacman_moves].any(axis=1)
            next_survivable &= self.maximin_strategy['safe']
            lost = response_table['alive'] & ~next_survivable
            depth = response_table['depth'] + 1
            response_table['ranks'][lost] = depth
            response_table['alive'], response_table['depth'] = next_survivable, depth
            response_table['solved'] = not lost.any()
        self._check_time()
        if (not response_table['solved']
                and response_table['ranks'][ghost_index, pacman_index] < 0
                and response_table['depth'] < remaining):
            return None
        predicted_pacman_index = response_table['policy'][ghost_index, pacman_index]
        best_move, best_score = None, None
        for move, pos in actions:
            self._check_time()
            rank = int(response_table['ranks'][index[pos], predicted_pacman_index])
            lifetime = remaining if rank < 0 else min(remaining, rank)
            safe = all(self._manhattan(pos, enemy) >= self.capture_distance
                       for enemy in self.pacman_targets[pacman])
            score = (lifetime, safe, self._evaluate(pos, pacman),
                     -self.visits[pos], move != Move.STAY)
            if best_score is None or score > best_score:
                best_move, best_score = move, score
        return best_move

    def _pacman_distance_map(self, start):
        # Trả {ô: số lượt Pacman ít nhất từ start}, có tính tốc độ và tường; dùng cache.
        if start not in self.pacman_turn_cache:
            self.pacman_turn_cache[start] = self._breadth_first_distances(
                (start,), pacman=True)
        return self.pacman_turn_cache[start]

    def _update_belief(self, grid, enemy, step_number):
        # Cập nhật tập ô Pacman có thể ở, rồi cập nhật xác suất trên tập đó.
        # enemy=None nghĩa là mất dấu; tập ô được mở rộng theo số lượt đã trôi qua.
        if enemy is not None:
            enemies = {enemy}
        else:
            elapsed = (step_number - self.previous_step
                       if self.previous_step is not None else 0)
            enemies = self.possible_enemies
            if not enemies or elapsed < 0:
                enemies = set(self.ghost_actions)
            for _ in range(min(max(0, elapsed), len(self.ghost_actions))):
                self._check_time()
                expanded = self._expand_enemies(enemies)
                if expanded == enemies:
                    break
                enemies = expanded
            enemies = {pos for pos in enemies if grid[pos] == -1}
            if not enemies:
                enemies = {pos for pos in self.ghost_actions if grid[pos] == -1}
        self._check_time()
        self.possible_enemies = enemies
        self.previous_step = step_number
        self._update_probabilities(enemy, step_number)

    def _prepare_belief(self):
        # Khởi tạo chỉ số ô, đích di chuyển, mặt nạ bắt và cache phục vụ suy đoán Pacman.
        if self.belief is not None or len(self.ghost_actions) > MAX_TABLE_CELLS:
            return
        nodes = sorted(self.ghost_actions)
        index = {pos: i for i, pos in enumerate(nodes)}
        width = max(map(len, self.pacman_targets.values()))
        edges = []
        for pos in nodes:
            self._check_time()
            targets = [index[nxt] for nxt in self.pacman_targets[pos]]
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
        # Cập nhật xác suất vị trí và trọng số kiểu đuổi; dùng lịch sử khi enemy=None.
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
                # Chỉ học từ hai lần thấy liên tiếp, để biết đúng một chuyển động.
                # observer là vị trí Ghost lượt trước vì hai bên di chuyển đồng thời.
                if previous_step + 1 == step_number:
                    models = self._pursuit_models(observer)
                    predicted = min(
                        self._ordered_pacman_targets(previous),
                        key=lambda pos: self._distance_map(observer).get(
                            pos, UNREACHABLE_DISTANCE))
                    bfs_consistent = bfs_consistent and predicted == enemy
                    bfs_observations = bfs_observations + 1 if predicted == enemy else 0
                    outcomes = belief['edges'][index[previous]] == index[enemy]
                    likelihood = models[:, index[previous], outcomes].sum(axis=1)
                    if likelihood.sum() > 0:
                        # Giữ một ít prior để mô hình tạm chưa khớp vẫn có thể phục hồi.
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
            # Loại các ô không còn khả thi theo quan sát và khả năng di chuyển.
            # Thêm xác suất nhỏ trên các ô còn hợp lệ trước khi chuẩn hóa về tổng 1.
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
        # Trả ma trận [ô Pacman, lựa chọn đích], trộn các kiểu đuổi nhắm vị trí ghost.
        belief = self.belief
        if ghost in belief['policies']:
            return belief['policies'][ghost]
        models = self._pursuit_models(ghost)
        weights = np.tensordot(self.model_weights, models, axes=(0, 0))
        self._check_time()
        belief['policies'][ghost] = weights
        return weights

    def _pursuit_models(self, ghost):
        # Tạo bốn kiểu đuổi nhắm ghost: BFS, ưu tiên dọc, ưu tiên ngang và ngẫu nhiên.
        belief = self.belief
        if ghost in belief['models']:
            return belief['models'][ghost]
        # models[kiểu đối thủ, ô Pacman hiện tại, lựa chọn đích] là xác suất chuyển ô.
        models = np.zeros((len(self.model_prior),) + belief['edges'].shape)
        distances = self._distance_map(ghost)
        for pacman_index, pacman in enumerate(belief['nodes']):
            self._check_time()
            targets = self.pacman_targets[pacman]
            costs = [distances.get(pos, UNREACHABLE_DISTANCE) for pos in targets]
            closest = min(costs)
            count = costs.count(closest)
            for endpoint_index, cost in enumerate(costs):
                models[MODEL_BFS, pacman_index, endpoint_index] = (cost == closest) / count
                models[MODEL_RANDOM, pacman_index, endpoint_index] = 1 / len(targets)
            dr, dc = ghost[0] - pacman[0], ghost[1] - pacman[1]
            vertical = Move.DOWN if dr > 0 else Move.UP if dr < 0 else None
            horizontal = Move.RIGHT if dc > 0 else Move.LEFT if dc < 0 else None
            preferences = ((vertical, horizontal), (horizontal, vertical))
            for model, preferred in enumerate(preferences, MODEL_GREEDY_VERTICAL):
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
                        if nxt not in self.ghost_actions:
                            break
                        pos = nxt
                    if pos != pacman:
                        selected = pos
                        break
                if selected is not None:
                    models[model, pacman_index, targets.index(selected)] = 1
                else:
                    endpoints = []
                    for direction in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                        pos = pacman
                        for _ in range(self.pacman_speed):
                            rr, cc = direction.value
                            nxt = (pos[0] + rr, pos[1] + cc)
                            if nxt not in self.ghost_actions:
                                break
                            pos = nxt
                        if pos != pacman:
                            endpoints.append(pos)
                    if not endpoints:
                        endpoints = [pacman]
                    for pos in endpoints:
                        models[model, pacman_index, targets.index(pos)] += 1 / len(endpoints)
        self._check_time()
        belief['models'][ghost] = models
        return models

    def _model_exploitable(self):
        # Trả True khi đủ quan sát và một kiểu greedy chiếm trọng số cao để dự đoán.
        greedy_weight = max(self.model_weights[MODEL_GREEDY_VERTICAL],
                            self.model_weights[MODEL_GREEDY_HORIZONTAL])
        return (self.model_observations >= MIN_MODEL_OBSERVATIONS
                and greedy_weight >= MODEL_CONFIDENCE_THRESHOLD)

    def _visible_from(self, pos):
        # Trả các ô thấy từ pos theo hình chữ thập, giới hạn bởi tường và vision_radius.
        key = (pos, self.vision_radius)
        if key not in self.visibility_cache:
            cells = {pos}
            for direction in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                dr, dc = direction.value
                r, c = pos
                for _ in range(self.vision_radius):
                    r, c = r + dr, c + dc
                    if (r, c) not in self.ghost_actions:
                        break
                    cells.add((r, c))
            self.visibility_cache[key] = tuple(cells)
        return self.visibility_cache[key]

    def _information_gain(self, pos, probabilities):
        # Tính thông tin kỳ vọng (bit) của quan sát tại pos từ phân phối probabilities.
        # Trả 0 khi không có fog hoặc không còn khối xác suất.
        mass = probabilities.sum()
        if not self.fog_enabled or mass <= 0:
            return 0
        index = self.belief['index']
        visible = probabilities[[index[cell] for cell in self._visible_from(pos)]] / mass
        outside = max(0, 1 - visible.sum())
        # Mỗi ô thấy được là một kết quả riêng; mọi ô ngoài tầm nhìn gộp thành "không thấy".
        # Entropy đo lượng thông tin kỳ vọng thu được nếu đứng ở pos.
        outcomes = np.append(visible[visible > 0], outside)
        outcomes = outcomes[outcomes > 0]
        return float(-(outcomes * np.log2(outcomes)).sum())

    def _turn_geometry(self, pos):
        # Trả hai vector khoảng cách/lượt chậm thêm và cờ góc rẽ để chấm điểm pos.
        # Lượt chậm thêm được so với đi thẳng ở tốc độ tối đa của Pacman.
        geometry = self.belief['geometry']
        if pos not in geometry:
            distances = self._distance_map(pos)
            turns = self._pacman_distance_map(pos)
            space, delay = [], []
            for enemy in self.belief['nodes']:
                distance = distances.get(enemy, UNREACHABLE_DISTANCE)
                space.append(min(40, distance))
                ideal = (distance + self.pacman_speed - 1) // self.pacman_speed
                delay.append(min(4, max(0, turns.get(enemy, UNREACHABLE_DISTANCE) - ideal))
                             if distance < UNREACHABLE_DISTANCE else 0)
            directions = {move for move, _ in self.ghost_actions[pos] if move != Move.STAY}
            corner = (self.pacman_speed > 1
                      and bool(directions & {Move.UP, Move.DOWN})
                      and bool(directions & {Move.LEFT, Move.RIGHT}))
            self._check_time()
            geometry[pos] = (np.asarray(space), np.asarray(delay), int(corner))
        return geometry[pos]

    def _forecast(self, probabilities, ghost):
        # Dự đoán phân phối Pacman sau một lượt khi Ghost ở ghost; trả vector xác suất mới.
        self._check_time()
        belief = self.belief
        # Nhiều ô xuất phát có thể dẫn tới cùng ô đích; bincount cộng xác suất về ô đó.
        return np.bincount(
            belief['edges'].ravel(),
            weights=(probabilities[:, None] * self._pursuit_policy(ghost)).ravel(),
            minlength=len(belief['nodes']))

    def _forecast_move(self, remaining, best_move, allowed=None):
        # Beam search chọn nước có xác suất sống sót cao và vị trí thuận lợi.
        # allowed giới hạn nước đầu tiên; best_move là dự phòng nếu hết giờ.
        belief = self.belief
        # Mỗi nhánh giữ nước đi đầu tiên để trả về sau khi xét sâu hơn.
        # Beam search chỉ giữ FORECAST_BEAM_WIDTH nhánh tốt nhất ở mỗi độ sâu.
        states = [(self.current_position, belief['probabilities'], None)]
        try:
            for depth in range(1, min(FORECAST_HORIZON, self.max_depth, remaining) + 1):
                candidates = []
                for ghost, probabilities, first in states:
                    prediction = self._forecast(probabilities, ghost)
                    for move, pos in self.ghost_actions[ghost]:
                        if depth == 1 and allowed is not None and move not in allowed:
                            continue
                        self._check_time()
                        following = prediction.copy()
                        following[belief['capture'][belief['index'][pos]]] = 0
                        # Không chuẩn hóa lại sau khi bỏ các trường hợp bị bắt.
                        # mass là xác suất sống sót suốt chuỗi nước đi đang xét.
                        mass = following.sum()
                        distances, turns, corner = self._turn_geometry(pos)
                        space = float(following @ distances)
                        information = self._information_gain(pos, following)
                        turn_delay = float(following @ turns)
                        utility = (10000 * mass + 0.4 * space
                                   + 0.25 * mass * information + 0.3 * turn_delay
                                   + 0.2 * mass * corner
                                   + mass * (len(self.ghost_actions[pos]) - 1
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
                          for _, first, pos, probabilities in unique[:FORECAST_BEAM_WIDTH]]
        except SearchTimeout:
            pass
        return best_move

    def _distance_map(self, start):
        # Trả {ô: số bước ngắn nhất từ start} và lưu cache; bỏ các ô không tới được.
        if start not in self.maze_distance_cache:
            self.maze_distance_cache[start] = self._breadth_first_distances((start,))
        return self.maze_distance_cache[start]

    def _expand_enemies(self, enemies):
        # Trả tập đích Pacman có thể tới sau một lượt từ bất kỳ ô nào trong enemies.
        expanded = set()
        for pos in enemies:
            self._check_time()
            expanded.update(self.pacman_targets[pos])
        return expanded

    def _breadth_first_distances(self, starts, pacman=False):
        # BFS nhiều nguồn: trả số cạnh ít nhất từ starts tới từng ô có thể tới.
        # pacman=True dùng nước đi tối đa pacman_speed ô làm một cạnh (một lượt).
        distances = dict.fromkeys(starts, 0)
        queue = deque(starts)
        while queue:
            self._check_time()
            pos = queue.popleft()
            neighbors = (self.pacman_targets[pos] if pacman else
                         (nxt for _, nxt in self.ghost_actions[pos]))
            for nxt in neighbors:
                if nxt not in distances:
                    distances[nxt] = distances[pos] + 1
                    queue.append(nxt)
        return distances

    def _safety_score(self, ghost, distance, pacman_turns):
        # Chấm vị trí Ghost từ khoảng cách, số lượt Pacman tới, lối ra và nguy cơ mắc kẹt.
        exits = len(self.ghost_actions[ghost]) - 1
        turns_to_reach = pacman_turns.get(ghost, UNREACHABLE_DISTANCE)
        return (12 * min(distance, 40) + 10 * min(turns_to_reach, 20) + 4 * exits
                - (24 if exits <= 1 else 0)
                - (32 if ghost in self.trapped else 0)
                - self._escape_penalty(ghost, pacman_turns))

    def _hidden_evaluate(self, ghost, distances, turns):
        # Chấm điểm khi mất dấu bằng khoảng cách/số lượt từ tập vị trí Pacman khả dĩ.
        return self._safety_score(ghost, distances.get(ghost, UNREACHABLE_DISTANCE), turns)

    def _hidden_value(self, ghost, level, depth, remaining,
                      hazards, distances, turns, table):
        # Trả điểm tốt nhất từ ghost ở lớp level, với vùng nguy hiểm hazards theo từng lượt.
        # Cache (ghost, level) chỉ dùng trong một lần tìm có depth và remaining cố định.
        self._check_time()
        key = (ghost, level)
        if key not in table:
            if ghost in hazards[level]:
                value = -TERMINAL_SCORE - remaining + level
            elif level + 1 >= remaining:
                value = TERMINAL_SCORE
            elif level + 1 == depth:
                value = self._hidden_evaluate(ghost, distances, turns)
            else:
                value = max(self._hidden_value(
                    nxt, level + 1, depth, remaining,
                    hazards, distances, turns, table)
                    for _, nxt in self.ghost_actions[ghost])
            table[key] = value
        return table[key]

    def _hidden_move(self, actions, remaining, best_move):
        # Khi mất dấu, né các vùng nguy hiểm dự báo; gần nguy hiểm thì dùng mô hình xác suất.
        # Giữ best_move nếu chưa hoàn thành lần tìm sâu tiếp theo.
        self.decision_mode = 'robust_hidden'
        if self.belief is not None and self.belief['probabilities'] is not None:
            distances = self._distance_map(self.current_position)
            nearest = min(
                (distances.get(pos, UNREACHABLE_DISTANCE) for pos in self.possible_enemies),
                default=UNREACHABLE_DISTANCE)
            if nearest <= self.pacman_speed + self.capture_distance - 1:
                self.decision_mode = 'forecast'
                return self._forecast_move(remaining, best_move)
        enemies = self.possible_enemies
        # hazards[d] gộp vùng bắt của mọi vị trí Pacman có thể tới sau d + 1 lượt.
        # Đây là dự báo bi quan khi chưa biết vị trí thật của Pacman.
        hazards = []
        try:
            for depth in range(1, min(self.max_depth, remaining) + 1):
                enemies = self._expand_enemies(enemies)
                danger = Counter()
                for pos in enemies:
                    self._check_time()
                    danger.update(self.capture_zones[pos])
                hazards.append(danger)
                distances = self._breadth_first_distances(enemies)
                turns = self._breadth_first_distances(enemies, pacman=True)
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
                if best_score < -TERMINAL_SCORE or best_score == TERMINAL_SCORE:
                    break
        except SearchTimeout:
            pass
        return best_move

    def _choose_fallback_move(self, grid, ghost, enemy):
        # Chọn nhanh Move hợp lệ, ưu tiên né vùng bắt của Pacman hoặc tới ô ít ghé.
        # Không cần bảng tiền xử lý, nên dùng được cả khi hết giờ ngay đầu lượt.
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
        # Chấm điểm cặp vị trí Ghost/Pacman; điểm cao nghĩa là Ghost thuận lợi hơn.
        distance = self._distance_map(pacman).get(ghost, UNREACHABLE_DISTANCE)
        pacman_distances = self._pacman_distance_map(pacman)
        return (self._safety_score(ghost, distance, pacman_distances)
                + 0.5 * min(self._manhattan(ghost, pacman), 20))

    def _check_time(self):
        # Ném SearchTimeout khi hết giờ để bên gọi dùng phương án đã hoàn thành gần nhất.
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def _maximin_search(self, ghost, pacman, depth, remaining, alpha, beta):
        # Tìm điểm Ghost bảo đảm ở độ sâu depth, với remaining lượt còn lại của trận.
        # alpha/beta là các cận cắt tỉa; trả điểm số từ góc nhìn Ghost.
        self._check_time()
        if remaining <= 0:
            return TERMINAL_SCORE
        if depth == 0:
            return self._evaluate(ghost, pacman)
        key = (ghost, pacman, depth, remaining)
        original_alpha, original_beta = alpha, beta
        cached = self.search_cache.get(key)
        if cached is not None:
            # Nhánh bị cắt tỉa chỉ cho cận dưới/trên; chỉ 'exact' là giá trị chính xác.
            # Dùng cận để thu hẹp alpha/beta trước khi quyết định có cần tìm tiếp.
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
        actions = sorted(self.ghost_actions[ghost],
                         key=lambda a: self._evaluate(a[1], pacman), reverse=True)
        for _, nxt in actions:
            value = max(value, self._worst_pacman_response(
                nxt, pacman, depth, remaining, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        flag = ('upper' if value <= original_alpha else
                'lower' if value >= original_beta else 'exact')
        self.search_cache[key] = (value, flag)
        return value

    def _worst_pacman_response(self, next_ghost, pacman, depth, remaining, alpha, beta):
        # Giữ nước Ghost đã chọn (next_ghost), lấy điểm xấu nhất trên mọi đích Pacman.
        # Cắt nhánh khi điểm không thể cải thiện lựa chọn Ghost đang có (alpha).
        # Simultaneous moves from old positions, capture only at endpoints.
        worst = float('inf')
        targets = sorted(self.pacman_targets[pacman],
                         key=lambda p: self._manhattan(next_ghost, p))
        for next_pacman in targets:
            self._check_time()
            if self._manhattan(next_ghost, next_pacman) < self.capture_distance:
                score = -TERMINAL_SCORE - remaining  # Delay unavoidable capture.
            else:
                score = self._maximin_search(
                    next_ghost, next_pacman, depth - 1,
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
            self._reset_episode_state()
        best_move = self._choose_fallback_move(map_state, my_position, enemy_position)
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
            actions = self.ghost_actions[my_position]
            remaining = max(1, self.max_steps - step_number + 1)
            if enemy_position is None and self.possible_enemies:
                return self._hidden_move(actions, remaining, best_move)
            if threat is None:
                # No information about enemy - explore less visited positions.
                return max(actions, key=lambda a: (
                    -self.visits[a[1]], len(self.ghost_actions[a[1]]), a[0] != Move.STAY))[0]

            # Under fog, the last sighting is only an estimate, not a guarantee.
            def fallback(action):
                # Xếp nước dự phòng theo an toàn lượt tới, điểm vị trí và số lần ghé.
                move, pos = action
                safety = min(self._manhattan(pos, p) for p in self.pacman_targets[threat])
                return (safety >= self.capture_distance, self._evaluate(pos, threat),
                        -self.visits[pos], move != Move.STAY)

            actions = sorted(actions, key=fallback, reverse=True)
            best_move = actions[0][0]
            if enemy_position is not None:
                strategy_move = self._choose_visible_move(
                    actions, my_position, enemy_position, remaining)
                if strategy_move is not None:
                    return strategy_move
            self.search_cache = {}
            # Iterative deepening preserves the last fully completed decision.
            for depth in range(1, min(self.max_depth, remaining) + 1):
                best_score, candidate = -float('inf'), best_move
                ordered = sorted(actions, key=lambda a: a[0] != best_move)
                for move, pos in ordered:
                    score = self._worst_pacman_response(
                        pos, threat, depth, remaining, best_score, float('inf'))
                    if score > best_score:
                        best_score, candidate = score, move
                best_move = candidate
        except SearchTimeout:
            pass
        return best_move
