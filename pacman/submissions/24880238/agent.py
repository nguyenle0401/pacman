"""Ghost: exact survival tables, online pursuit models and bounded fog search."""
import heapq
import sys
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path
import numpy as np

# Add src to path to import the interface
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


MAX_TABLE_BYTES = 64 * 1024 * 1024
MAX_BELIEF_CELLS = 2048
MODEL_CACHE_SIZE = 64
DISTANCE_CACHE_SIZE = 128
MAX_ESCAPE_REGION = 64
UNREACHABLE_DISTANCE = 1000
TERMINAL_SCORE = 100000
STRATEGY_TIME_FRACTION = 0.65
MODEL_BFS = 0
MODEL_GREEDY_VERTICAL = 1
MODEL_GREEDY_HORIZONTAL = 2
MODEL_RANDOM = 3
MODEL_BFS_RANDOM = 4
MODEL_BFS_REVERSE = 5
MODEL_BFS_SLOW = 6
MODEL_INTERCEPT = 7
RESPONSE_MODELS = (MODEL_BFS, MODEL_BFS_REVERSE, MODEL_BFS_SLOW, MODEL_INTERCEPT,
                   MODEL_GREEDY_VERTICAL, MODEL_GREEDY_HORIZONTAL)
MIN_MODEL_OBSERVATIONS = 2
MODEL_CONFIDENCE_THRESHOLD = 0.55
FORECAST_HORIZON = 4
FORECAST_BEAM_WIDTH = 8
CAPTURE_MODEL_MATCHES = 3
CAPTURE_CYCLE_DEPTH = 12
CAPTURE_CYCLE_NODES = 1500
CAPTURE_CACHE_SIZE = 512


class SearchTimeout(Exception):
    pass


class GhostAgent(BaseGhostAgent):
    """Evade simultaneous moves using robust and learned survival tables.

    Fog decisions use a filtered position belief and a bounded beam search.
    Large maps fall back to iterative deepening when pair tables do not fit.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ghost receives no settings from the loader. These match CLI defaults.
        self.pacman_speed = max(1, int(kwargs.get('pacman_speed', 2)))
        self.capture_distance = max(1, int(kwargs.get('capture_distance', 2)))
        self.max_steps = int(kwargs.get('max_steps', 200))
        self.time_budget = max(0.0, float(kwargs.get('time_budget', 0.15)))
        self.max_depth = max(1, int(kwargs.get('max_depth', 8)))
        self.forecast_horizon = max(1, int(kwargs.get('forecast_horizon', FORECAST_HORIZON)))
        self.forecast_beam_width = max(5, int(kwargs.get('forecast_beam_width', FORECAST_BEAM_WIDTH)))
        self.policy_tiebreak = bool(kwargs.get('policy_tiebreak', True))
        self.learned_tables = bool(kwargs.get('learned_tables', True))
        # Exact belief identities are available for experiments, but the compact
        # position beam preserves stronger spatial coverage at the default width.
        self.belief_merge = kwargs.get('belief_merge', 'position')
        if self.belief_merge not in ('exact', 'position'):
            raise ValueError("belief_merge must be 'exact' or 'position'")
        self.opponent_memory = bool(kwargs.get('opponent_memory', True))
        # Long waits in narrow corridors can hide an approaching enemy until
        # capture. Zero retains the unconditional Round 3 memory model.
        self.memory_stall_limit = max(0, int(kwargs.get('memory_stall_limit', 0)))
        self.memory_full_prior = min(1.0, max(0.0, float(kwargs.get('memory_full_prior', 0.5))))
        self.search_exploration = max(0.0, float(kwargs.get('search_exploration', 1.0)))
        self.stochastic_survival = bool(kwargs.get('stochastic_survival', False))
        self.capture_cycles = bool(kwargs.get('capture_cycles', True))
        self.predictive_cycles = bool(kwargs.get('predictive_cycles', True))
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.visits = Counter()
        self.topology = None
        self.possible_enemies = set()
        self.deadline = float('inf')
        self.maximin_strategy = None
        self.current_position = None
        self.model_prior = np.asarray([0.28, 0.22, 0.22, 0.07, 0.07, 0.05, 0.03, 0.06])
        # Before any observed transition, spread BFS ties rather than predicting
        # one arbitrary tie-break throughout an unseen opponent's trajectory.
        self.fog_prior = np.zeros(len(self.model_prior))
        self.fog_prior[[MODEL_BFS_RANDOM, MODEL_GREEDY_VERTICAL,
                        MODEL_GREEDY_HORIZONTAL, MODEL_RANDOM]] = (0.4, 0.25, 0.25, 0.1)
        self.fog_enabled = False
        self.decision_mode = 'fallback'
        self.policy_strategy = None
        self.policy_strategies = OrderedDict()
        self.policy_builders = {}
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
        self.model_streaks = np.zeros(len(self.model_prior), dtype=np.int32)
        self.vision_radius = 0
        self.visibility_cache = {}
        self.response_model = MODEL_BFS
        self.memory_stall_steps = 0
        self.memory_observer = None
        self.memory_fallback = False
        self.capture_streak = 0
        self.capture_last_sighting = None
        self.capture_plan = {}
        self.capture_cache = OrderedDict()
        self.predictive_last_sighting = None
        self.predictive_heading = (0, 0)
        self.predictive_streak = 0
        self.predictive_distinct = False
        self.predictive_plan = {}
        self.predictive_cache = OrderedDict()

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
        full_deadline = self.deadline
        now = time.perf_counter()
        self.deadline = min(full_deadline, now + max(0, full_deadline - now) * 0.4)
        try:
            escape_routes = self._find_escape_routes(all_actions)
        except SearchTimeout:
            # Optional pocket analysis must not prevent the cheap topology from
            # ever being published on a large map. Dead-end detection is retained.
            escape_routes = {pos: () for pos in cells}
        finally:
            self.deadline = full_deadline
        self.topology = topology
        self.ghost_actions, self.pacman_targets = all_actions, all_targets
        self.ghost_neighbors = {pos: tuple(nxt for move, nxt in moves if move != Move.STAY)
                                for pos, moves in all_actions.items()}
        self.capture_zones, self.trapped = capture_zones, trapped
        self.escape_routes = escape_routes
        self.maze_distance_cache, self.pacman_turn_cache = OrderedDict(), OrderedDict()
        self.expansion_cache = OrderedDict()
        self.multi_distance_cache = OrderedDict()
        self.evaluation_cache = OrderedDict()
        self.maximin_strategy = None
        self.fog_enabled = False
        self.policy_strategy = None
        self.policy_strategies = OrderedDict()
        self.policy_builders = {}
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

        # Only small pockets materially contribute to the escape penalty. Avoid
        # traversing the entire map once per articulation point on large mazes.
        component_sizes = {}
        for root in neighbors:
            if root in component_sizes:
                continue
            component, queue = {root}, deque([root])
            while queue:
                pos = queue.popleft()
                for nxt in neighbors[pos]:
                    if nxt not in component:
                        component.add(nxt)
                        queue.append(nxt)
            component_sizes.update(dict.fromkeys(component, len(component)))
        routes = {pos: [] for pos in neighbors}
        for gate in sorted(gates):
            limit = min(MAX_ESCAPE_REGION, (component_sizes[gate] - 1) // 2)
            seen, regions = {gate}, []
            for start in neighbors[gate]:
                if start in seen:
                    continue
                region, queue = {start}, deque([start])
                while queue and len(region) <= limit:
                    self._check_time()
                    pos = queue.popleft()
                    for nxt in neighbors[pos]:
                        if nxt != gate and nxt not in region:
                            region.add(nxt)
                            queue.append(nxt)
                if not queue and len(region) <= limit:
                    seen.update(region)
                    regions.append(region)
            for region in regions:
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
        if self.maximin_strategy is not None:
            return
        size = len(self.ghost_actions)
        # Account for bounded policy tables and NumPy gather temporaries.
        width = max(map(len, self.pacman_targets.values()))
        if size * size * (80 + width) > MAX_TABLE_BYTES:
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
        coordinates = np.asarray(cells, dtype=np.int32)
        safe = (np.abs(coordinates[:, None, 0] - coordinates[None, :, 0])
                + np.abs(coordinates[:, None, 1] - coordinates[None, :, 1])
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
            self._advance_maximin_once()
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
            predicted = 0.0
            if self.policy_tiebreak and self.belief is not None:
                policy = self._pursuit_policy(ghost)[self.belief['index'][pacman]]
                unknown = remaining if strategy['solved'] else min(remaining, strategy['depth'] + 1)
                lifetimes = np.where(values < 0, unknown, np.minimum(values, remaining))
                predicted = float(lifetimes @ policy[:len(targets)])
            score = (outcomes[0], predicted, outcomes, self._evaluate(pos, pacman),
                     -self.visits[pos], move != Move.STAY)
            if best_score is None or score > best_score:
                best_move, best_score = move, score
        return best_move

    def _advance_maximin_once(self):
        strategy = self.maximin_strategy
        survivable = strategy['alive']
        safe_responses = survivable[:, strategy['pacman_moves']].all(axis=2)
        following = safe_responses[strategy['ghost_moves']].any(axis=1)
        following &= strategy['safe']
        lost = survivable & ~following
        depth = strategy['depth'] + 1
        strategy['ranks'][lost] = depth
        strategy['alive'], strategy['depth'] = following, depth
        strategy['solved'] = not lost.any()

    def _policy_model_supported(self):
        # A prior is not evidence. Bayesian weights can recover after a mismatch.
        candidates = RESPONSE_MODELS if self.learned_tables else RESPONSE_MODELS[:4]
        self.response_model = max(candidates,
                             key=lambda model: self.model_weights[model])
        weight = self.model_weights[self.response_model]
        matches = self.model_streaks[self.response_model]
        supported = (weight >= MODEL_CONFIDENCE_THRESHOLD
                     or (matches >= MIN_MODEL_OBSERVATIONS
                         and weight >= 0.95 * self.model_weights.max()))
        return ((self.model_observations >= MIN_MODEL_OBSERVATIONS and matches > 0 and supported)
                or (self.model_observations >= 1 and matches > 0 and weight >= 0.75))

    def _choose_visible_base_move(self, actions, ghost, pacman, remaining):
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
        # On maps where robust evasion is impossible, a common chase policy may
        # still admit an infinite escape cycle. Prefer that opening opportunity
        # only when its table has actually converged; this is a prior-based
        # choice, not evidence that the opponent follows that policy.
        opening_supported = (self.model_observations == 0
                             or (self.model_observations == 1 and self.model_streaks[MODEL_BFS] > 0
                                 and self.model_weights[MODEL_BFS] >= self.model_weights.max() * 0.95))
        if not self.fog_enabled and opening_supported:
            opening = self._choose_policy_survival_move(actions, ghost, pacman, remaining)
            table = self.policy_strategy
            if (opening is not None and table['solved']
                    and table['ranks'][index[ghost], index[pacman]] < 0):
                self.decision_mode = 'opening_survival'
                return opening
        if self._policy_model_supported():
            policy_move = self._choose_policy_survival_move(actions, ghost, pacman, remaining,
                                                          self.response_model)
            if policy_move is not None:
                self.decision_mode = 'policy_survival'
                return policy_move
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

    def _choose_capture_move(self, actions, ghost, pacman, remaining):
        move = self._choose_visible_base_move(actions, ghost, pacman, remaining)
        if (not self.capture_cycles or move is None or self.fog_enabled
                or self.vision_radius > 0 or self.maximin_strategy is None):
            self.capture_last_sighting = None
            self.capture_streak = 0
            self.capture_plan.clear()
            return move
        index = self.maximin_strategy['index']
        g, p = index[ghost], index[pacman]
        if not 0 < self.maximin_strategy['ranks'][g, p] <= remaining:
            return move
        previous = self.capture_last_sighting
        self.capture_last_sighting = (pacman, ghost, self.previous_step)
        try:
            if (previous is not None and previous[2] is not None
                    and self.previous_step == previous[2] + 1):
                predicted = self._capture_prediction(previous[1], previous[0])
                self.capture_streak = self.capture_streak + 1 if predicted == pacman else 0
            else:
                self.capture_streak = 0
            if not self.capture_streak:
                self.capture_plan.clear()
            if self.capture_streak < CAPTURE_MODEL_MATCHES:
                return move
            state = (ghost, pacman)
            if state in self.capture_plan:
                self.decision_mode = 'capture_cycle'
                return self.capture_plan[state]
            # Preserve a winning response against an already supported model
            # before introducing another hypothesis. Existing cycles continue
            # while every observed transition still agrees with their model.
            if self.decision_mode == 'policy_survival' and self.policy_strategy is not None:
                rank = self.policy_strategy['ranks'][g, p]
                if rank < 0 or rank > remaining:
                    return move
            plan = self._find_capture_cycle(ghost, pacman)
            if plan:
                self.capture_plan = plan
                self.decision_mode = 'capture_cycle'
                return plan[state]
        except SearchTimeout:
            # The previous decision was completed before this optional search.
            self.capture_streak = 0
            self.capture_plan.clear()
        return move

    def _choose_visible_move(self, actions, ghost, pacman, remaining):
        move = self._choose_capture_move(actions, ghost, pacman, remaining)
        if (not self.capture_cycles or not self.predictive_cycles or move is None
                or self.fog_enabled or self.vision_radius > 0 or self.maximin_strategy is None):
            self._reset_predictive_evidence()
            return move
        index = self.maximin_strategy['index']
        g, p = index[ghost], index[pacman]
        if not 0 < self.maximin_strategy['ranks'][g, p] <= remaining:
            return move
        previous = self.predictive_last_sighting
        previous_heading = self.predictive_heading
        self.predictive_last_sighting = (pacman, ghost, self.previous_step)
        try:
            if (previous is not None and previous[2] is not None
                    and self.previous_step == previous[2] + 1):
                predicted = self._predictive_prediction(previous[1], previous[0], previous_heading)
                self.predictive_heading = (ghost[0] - previous[1][0], ghost[1] - previous[1][1])
                if predicted == pacman:
                    self.predictive_streak += 1
                    # Require evidence distinguishing motion prediction from
                    # ordinary pursuit, rather than activating on a shared prefix.
                    self.predictive_distinct |= predicted != self._predictive_capture_prediction(previous[1], previous[0])
                else:
                    self.predictive_streak = 0
                    self.predictive_distinct = False
            else:
                self.predictive_heading = (0, 0)
                self.predictive_streak = 0
                self.predictive_distinct = False
            if not self.predictive_streak:
                self.predictive_plan.clear()
            if self.predictive_streak < CAPTURE_MODEL_MATCHES or not self.predictive_distinct:
                return move
            state = (ghost, pacman, self.predictive_heading)
            if state in self.predictive_plan:
                self.decision_mode = 'predictive_cycle'
                return self.predictive_plan[state]
            if self.decision_mode == 'capture_cycle':
                return move
            if self.decision_mode == 'policy_survival' and self.policy_strategy is not None:
                rank = self.policy_strategy['ranks'][g, p]
                if rank < 0 or rank > remaining:
                    return move
            plan = self._find_predictive_cycle(*state)
            if plan:
                self.predictive_plan = plan
                self.decision_mode = 'predictive_cycle'
                return plan[state]
        except SearchTimeout:
            self._reset_predictive_evidence()
        return move

    def _reset_predictive_evidence(self):
        self.predictive_last_sighting = None
        self.predictive_heading = (0, 0)
        self.predictive_streak = 0
        self.predictive_distinct = False
        self.predictive_plan.clear()

    def _predictive_prediction(self, ghost, pacman, heading):
        """Hypothesize an A* opponent aiming two Ghost moves ahead.

        The heading belongs to the observed state, before the simultaneous
        move. Extrapolation stops at walls and reuses the bounded A* cache.
        """
        target = ghost
        for _ in range(2):
            following = (target[0] + heading[0], target[1] + heading[1])
            if following not in self.ghost_actions:
                break
            target = following
        return self._predictive_capture_prediction(target, pacman)

    def _predictive_capture_prediction(self, target, pacman):
        """Use a separate cache so hypothesis testing leaves the old model alone."""
        key = (target, pacman)
        if key in self.predictive_cache:
            self.predictive_cache.move_to_end(key)
            return self.predictive_cache[key]

        def heuristic(pos):
            return max(0, self._manhattan(pos, target) - self.capture_distance
                       + self.pacman_speed) // self.pacman_speed

        h = heuristic(pacman)
        queue = [(h, h, 0, pacman, None)]
        costs, serial, iterations = {pacman: 0}, 0, 0
        result = pacman
        while queue:
            if iterations & 15 == 0:
                self._check_time()
            iterations += 1
            priority, h, _, cell, first = heapq.heappop(queue)
            if priority - h != costs[cell]:
                continue
            if self._manhattan(cell, target) < self.capture_distance:
                result = first if first is not None else pacman
                break
            cost = costs[cell] + 1
            targets = self.ordered_target_cache.get(cell)
            if targets is None:
                # Real model activation follows belief preparation, which has
                # already built these tuples. Keep direct helper calls pure too.
                targets = []
                for move in Move:
                    current = cell
                    dr, dc = move.value
                    for _ in range(min(self.pacman_speed, max(self.topology[0]))):
                        nxt = (current[0] + dr, current[1] + dc)
                        if nxt in self.ghost_actions:
                            current = nxt
                        targets.append(current)
            stride = len(targets) // len(Move)
            for direction in (3, 1, 2, 0):
                for step in range(stride - 1, -1, -1):
                    nxt = targets[direction * stride + step]
                    if cost < costs.get(nxt, float('inf')):
                        costs[nxt] = cost
                        serial += 1
                        nh = heuristic(nxt)
                        heapq.heappush(queue, (cost + nh, nh, serial, nxt,
                                               nxt if first is None else first))
        self.predictive_cache[key] = result
        if len(self.predictive_cache) > CAPTURE_CACHE_SIZE:
            self.predictive_cache.popitem(last=False)
        return result

    def _capture_prediction(self, ghost, pacman):
        """Predict a shortest-turn A* chaser, with deterministic tie ordering.

        This is an opponent hypothesis checked against visible transitions;
        it is not assumed to describe every shortest-path opponent.
        """
        key = (ghost, pacman)
        if key in self.capture_cache:
            self.capture_cache.move_to_end(key)
            return self.capture_cache[key]

        def heuristic(pos):
            distance = self._manhattan(pos, ghost)
            return max(0, distance - self.capture_distance + self.pacman_speed) // self.pacman_speed

        h = heuristic(pacman)
        queue = [(h, h, 0, pacman, None)]
        costs, serial, iterations = {pacman: 0}, 0, 0
        result = pacman
        while queue:
            if iterations & 15 == 0:
                self._check_time()
            iterations += 1
            priority, h, _, cell, first = heapq.heappop(queue)
            if priority - h != costs[cell]:
                continue
            if self._manhattan(cell, ghost) < self.capture_distance:
                result = first if first is not None else pacman
                break
            cost = costs[cell] + 1
            targets = self._ordered_pacman_targets(cell)
            stride = len(targets) // len(Move)
            for direction in (3, 1, 2, 0):  # Right, down, left, up; longer steps first.
                for step in range(stride - 1, -1, -1):
                    nxt = targets[direction * stride + step]
                    if cost < costs.get(nxt, float('inf')):
                        costs[nxt] = cost
                        serial += 1
                        nh = heuristic(nxt)
                        heapq.heappush(queue, (cost + nh, nh, serial, nxt,
                                               nxt if first is None else first))
        self.capture_cache[key] = result
        if len(self.capture_cache) > CAPTURE_CACHE_SIZE:
            self.capture_cache.popitem(last=False)
        return result

    def _find_capture_cycle(self, ghost, pacman):
        """Return a closed safe policy for the deterministic model, or no proof.

        A path back into the active DFS stack gives a finite prefix and cycle.
        Its guarantee is conditional on the model continuing to match; hidden
        observations and mismatches prevent its use.
        """
        path, actions, active, failed = [], [], set(), set()
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * 0.5
        visits = 0

        def search(state, left):
            nonlocal visits
            visits += 1
            if visits > CAPTURE_CYCLE_NODES or time.perf_counter() >= stop:
                return None
            self._check_time()
            if state in active:
                return dict(zip(path, actions))
            if not left or (state, left) in failed:
                return None
            g, p = state
            following = self._capture_prediction(g, p)
            choices = [(move, pos) for move, pos in self.ghost_actions[g]
                       if self._manhattan(pos, following) >= self.capture_distance]
            choices.sort(key=lambda action: (self._manhattan(action[1], following),
                                              len(self.ghost_actions[action[1]]),
                                              -self.visits[action[1]]), reverse=True)
            active.add(state)
            path.append(state)
            for move, pos in choices:
                actions.append(move)
                plan = search((pos, following), left - 1)
                if plan is not None:
                    return plan
                actions.pop()
            path.pop()
            active.remove(state)
            failed.add((state, left))
            return None

        return search((ghost, pacman), CAPTURE_CYCLE_DEPTH)

    def _find_predictive_cycle(self, ghost, pacman, heading):
        """A closed predictive policy must also return to the same heading."""
        path, actions, active, failed = [], [], set(), set()
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * 0.5
        visits = 0

        def search(state, left):
            nonlocal visits
            visits += 1
            if visits > CAPTURE_CYCLE_NODES or time.perf_counter() >= stop:
                return None
            self._check_time()
            if state in active:
                return dict(zip(path, actions))
            if not left or (state, left) in failed:
                return None
            g, p, h = state
            following = self._predictive_prediction(g, p, h)
            choices = [(move, pos) for move, pos in self.ghost_actions[g]
                       if self._manhattan(pos, following) >= self.capture_distance]
            choices.sort(key=lambda action: (self._manhattan(action[1], following),
                                              len(self.ghost_actions[action[1]]),
                                              -self.visits[action[1]]), reverse=True)
            active.add(state)
            path.append(state)
            for move, pos in choices:
                actions.append(move)
                successor = (pos, following, (pos[0] - g[0], pos[1] - g[1]))
                plan = search(successor, left - 1)
                if plan is not None:
                    return plan
                actions.pop()
            path.pop()
            active.remove(state)
            failed.add((state, left))
            return None

        return search((ghost, pacman, heading), CAPTURE_CYCLE_DEPTH)

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

    def _prepare_policy_strategy(self, model=MODEL_BFS):
        # Khởi tạo bảng nước đi BFS dự đoán và bảng sống sót cho mọi cặp vị trí.
        if model in self.policy_strategies:
            self.policy_strategies.move_to_end(model)
            self.policy_strategy = self.policy_strategies[model]
            return
        self._prepare_maximin_strategy()
        if self.maximin_strategy is None:
            return
        self._prepare_belief()
        if self.belief is None:
            return
        self.policy_strategy = None
        nodes = self.belief['nodes']
        index = self.maximin_strategy['index']
        size = len(nodes)
        greedy = model in (MODEL_GREEDY_VERTICAL, MODEL_GREEDY_HORIZONTAL)
        policy_width = 4 if greedy else 1
        if model not in self.policy_builders:
            # Only the currently selected model needs an unfinished allocation.
            self.policy_builders.clear()
            self.policy_builders[model] = dict(policy=np.empty((size, size, policy_width), dtype=np.int32), built=0)
        builder = self.policy_builders[model]
        rows = np.arange(size)
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * STRATEGY_TIME_FRACTION
        while builder['built'] < size:
            if time.perf_counter() >= stop:
                return
            row = builder['built']
            ghost = nodes[row]
            if greedy:
                end = min(size, row + 64)
                builder['policy'][row:end] = self._greedy_policy_chunk(row, end, model)
                builder['built'] = end
                continue
            else:
                ordered = self.belief['ordered']
                if model == MODEL_BFS_REVERSE:
                    ordered = ordered[:, ::-1]
                elif model == MODEL_BFS_SLOW:
                    ordered = ordered[:, ::ordered.shape[1] // len(Move)]
                distances = self._distance_map(ghost)
                costs = np.asarray([distances.get(pos, UNREACHABLE_DISTANCE) for pos in nodes])
                builder['policy'][row, :, 0] = (self._intercept_targets(ghost, costs)
                    if model == MODEL_INTERCEPT else ordered[rows, costs[ordered].argmin(axis=1)])
            builder['built'] += 1
        policy = builder['policy']
        safe = self.maximin_strategy['safe']
        ranks = np.full(safe.shape, -1, dtype=np.int32)
        ranks[~safe] = 0
        self._check_time()
        self.policy_strategy = {
            'index': index, 'policy': policy, 'ranks': ranks,
            'alive': safe.copy(), 'depth': 0, 'solved': False,
            'branches': np.where((policy != policy[:, :, :1]).any(axis=2)),
        }
        self.policy_strategies[model] = self.policy_strategy
        if len(self.policy_strategies) > 2:
            self.policy_strategies.popitem(last=False)
        del self.policy_builders[model]

    def _greedy_policy_chunk(self, first, end, model):
        """All greedy responses for a block of Ghost cells, without any BFS."""
        belief = self.belief
        coordinates = belief['coordinates']
        size = len(coordinates)
        rows = np.arange(size)[None, :]
        stride = belief['ordered'].shape[1] // len(Move)
        directional = belief['ordered'].reshape(size, len(Move), stride)
        dr = coordinates[first:end, None, 0] - coordinates[None, :, 0]
        dc = coordinates[first:end, None, 1] - coordinates[None, :, 1]
        vertical = directional[rows, np.where(dr > 0, 1, 0),
                               np.maximum(0, np.minimum(np.abs(dr), stride) - 1)]
        horizontal = directional[rows, np.where(dc > 0, 3, 2),
                                 np.maximum(0, np.minimum(np.abs(dc), stride) - 1)]
        vertical = np.where(dr != 0, vertical, rows)
        horizontal = np.where(dc != 0, horizontal, rows)
        primary, secondary = ((vertical, horizontal) if model == MODEL_GREEDY_VERTICAL
                              else (horizontal, vertical))
        selected = np.where(primary != rows, primary, secondary)
        fallback = directional[:, :4, -1]
        moving = fallback != np.arange(size)[:, None]
        first_fallback = fallback[np.arange(size), moving.argmax(axis=1)]
        fallback = np.where(moving, fallback, first_fallback[:, None])
        return np.where((selected != rows)[:, :, None], selected[:, :, None], fallback[None, :, :])

    def _choose_policy_survival_move(self, actions, ghost, pacman, remaining, model=MODEL_BFS):
        # Chọn Move kéo dài thời gian sống nhất trước chính sách BFS giả định.
        # Trả None nếu bảng chưa tính đủ để quyết định.
        self._prepare_policy_strategy(model)
        response_table = self.policy_strategy
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
            # First evaluate the deterministic response, then refine only states
            # with multiple supported responses (e.g. random greedy fallbacks).
            ghost_moves = self.maximin_strategy['ghost_moves']
            policy = response_table['policy']
            responses = response_table['alive'][ghost_moves[:, :, None], policy[:, None, :, 0]]
            branch_ghosts, branch_pacmen = response_table['branches']
            if len(branch_ghosts):
                alternatives = response_table['alive'][
                    ghost_moves[branch_ghosts, :, None],
                    policy[branch_ghosts, branch_pacmen][:, None, :]]
                responses[branch_ghosts, :, branch_pacmen] = alternatives.all(axis=2)
            next_survivable = responses.any(axis=1)
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
        predicted_pacman_indices = response_table['policy'][ghost_index, pacman_index]
        chance = None
        capture_turn = response_table['ranks'][ghost_index, pacman_index]
        if (self.stochastic_survival and model in (MODEL_GREEDY_VERTICAL, MODEL_GREEDY_HORIZONTAL)
                and 0 < capture_turn <= remaining):
            values = self._stochastic_values(response_table, remaining - 1)
            if values is not None:
                probabilities = response_table['chance']['weights'][ghost_index, pacman_index]
                chance = {move: float(values[index[pos], predicted_pacman_indices] @ probabilities)
                          for move, pos in actions}
                if max(chance.values(), default=0) <= 1e-12:
                    chance = None
        best_move, best_score = None, None
        for move, pos in actions:
            self._check_time()
            ranks = response_table['ranks'][index[pos], predicted_pacman_indices]
            lifetime = min(remaining if rank < 0 else min(remaining, int(rank)) for rank in ranks)
            safe = all(self._manhattan(pos, enemy) >= self.capture_distance
                       for enemy in self.pacman_targets[pacman])
            score = ((chance[move] if chance is not None else 0), lifetime, safe, self._evaluate(pos, pacman),
                     -self.visits[pos], move != Move.STAY)
            if best_score is None or score > best_score:
                best_move, best_score = move, score
        return best_move

    def _stochastic_values(self, table, turns):
        """Finite-horizon survival probability for a learned greedy policy.

        All fallback directions have equal probability. A stable Bellman table
        is exact for every longer horizon; otherwise only completed horizons
        are returned. Guaranteed wins remain handled by the robust tables.
        """
        size = len(table['index'])
        policy = table['policy']
        if size * size * 8 * (33 + policy.shape[2]) > 16 * 1024 * 1024:
            return None
        if 'chance' not in table:
            unique = np.ones(policy.shape, dtype=bool)
            for k in range(1, policy.shape[2]):
                unique[:, :, k] = (policy[:, :, k, None] != policy[:, :, :k]).all(axis=2)
            weights = unique / unique.sum(axis=2, keepdims=True)
            table['chance'] = dict(weights=weights, values=[self.maximin_strategy['safe'].astype(float)],
                                   converged=False)
        chance = table['chance']
        now = time.perf_counter()
        stop = now + max(0, self.deadline - now) * 0.65
        ghosts = self.maximin_strategy['ghost_moves']
        while (len(chance['values']) <= min(turns, 32) and not chance['converged']
               and time.perf_counter() < stop):
            previous = chance['values'][-1]
            following = np.zeros_like(previous)
            for action in range(ghosts.shape[1]):
                expected = (previous[ghosts[:, action, None, None], policy] * chance['weights']).sum(axis=2)
                np.maximum(following, expected, out=following)
            following *= self.maximin_strategy['safe']
            chance['converged'] = np.array_equal(previous, following)
            chance['values'].append(following)
        self._check_time()
        if turns < len(chance['values']):
            return chance['values'][turns]
        return chance['values'][-1] if chance['converged'] else None

    def _pacman_distance_map(self, start):
        # Trả {ô: số lượt Pacman ít nhất từ start}, có tính tốc độ và tường; dùng cache.
        if start not in self.pacman_turn_cache:
            self.pacman_turn_cache[start] = self._breadth_first_distances(
                (start,), pacman=True)
            if len(self.pacman_turn_cache) > DISTANCE_CACHE_SIZE:
                self.pacman_turn_cache.popitem(last=False)
        else:
            self.pacman_turn_cache.move_to_end(start)
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
        if self.belief is not None or len(self.ghost_actions) > MAX_BELIEF_CELLS:
            return
        nodes = sorted(self.ghost_actions)
        index = {pos: i for i, pos in enumerate(nodes)}
        width = max(map(len, self.pacman_targets.values()))
        edges = []
        for pos in nodes:
            self._check_time()
            targets = [index[nxt] for nxt in self.pacman_targets[pos]]
            edges.append(targets + [index[pos]] * (width - len(targets)))
        coordinates = np.asarray(nodes, dtype=np.int32)
        capture = (np.abs(coordinates[:, None, 0] - coordinates[None, :, 0])
                   + np.abs(coordinates[:, None, 1] - coordinates[None, :, 1])
                   < self.capture_distance)
        self._check_time()
        self.belief = {
            'nodes': nodes,
            'index': index,
            'edges': np.asarray(edges, dtype=np.int32),
            'capture': capture,
            'coordinates': coordinates,
            'counts': np.asarray([len(self.pacman_targets[pos]) for pos in nodes]),
            'ordered': np.asarray([[index[target] for target in self._ordered_pacman_targets(pos)]
                                   for pos in nodes], dtype=np.int32),
            'policies': OrderedDict(),
            'models': OrderedDict(),
            'geometry': OrderedDict(),
            'cache_limit': max(2, min(MODEL_CACHE_SIZE,
                                     (8 * 1024 * 1024) // (len(self.model_prior) * len(nodes) * width * 4))),
            'probabilities': None,
            'visibility_masks': OrderedDict(),
            'enemy_memories': None,
            'enemy_visits': np.zeros(len(nodes)),  # Expected occupancy inferred from observations.
            'search_policy': None,
            'observer': None,
            'step': None,
        }

    def _update_probabilities(self, enemy, step_number):
        # Cập nhật xác suất vị trí và trọng số kiểu đuổi; dùng lịch sử khi enemy=None.
        self._prepare_belief()
        belief = self.belief
        if belief is None:
            return
        self._update_memory_mode(enemy)
        nodes, index = belief['nodes'], belief['index']
        learned = None
        streaks = self.model_streaks
        if enemy is not None:
            if self.last_sighting is not None:
                previous, observer, previous_step = self.last_sighting
                # Chỉ học từ hai lần thấy liên tiếp, để biết đúng một chuyển động.
                # observer là vị trí Ghost lượt trước vì hai bên di chuyển đồng thời.
                if previous_step + 1 == step_number:
                    models = self._pursuit_models(observer)
                    outcomes = belief['edges'][index[previous]] == index[enemy]
                    likelihood = models[:, index[previous], outcomes].sum(axis=1)
                    streaks = np.where(likelihood > 0, np.minimum(streaks + 1, 1000), 0)
                    if likelihood.sum() > 0:
                        # Giữ một ít prior để mô hình tạm chưa khớp vẫn có thể phục hồi.
                        learned = self.model_weights * (likelihood + 0.04)
                        learned = 0.95 * learned / learned.sum() + 0.05 * self.model_prior
        memory_workspace = (len(nodes) * belief['edges'].shape[1]
                            * self.forecast_beam_width * 16 * 24)
        if (self.opponent_memory and not self.memory_fallback and self.fog_enabled
                and memory_workspace <= 32 * 1024 * 1024):
            self._update_enemy_memories(enemy, step_number, learned, streaks)
            return
        belief['enemy_memories'] = None
        if enemy is not None:
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
            # Being alive is also an observation, including diagonal capture zones.
            mask &= ~belief['capture'][index[self.current_position]]
            probabilities *= mask
            if probabilities.sum() <= 1e-12:
                probabilities = mask.astype(float)
            probabilities += mask * 1e-6
            mass = probabilities.sum()
            if mass <= 0:
                return
            probabilities /= mass
        self._check_time()
        if learned is not None:
            self.model_weights = learned
            self.model_streaks = streaks
            self.model_observations += 1
            belief['policies'].clear()
        belief['probabilities'] = probabilities
        belief['observer'] = self.current_position
        belief['step'] = step_number
        if enemy is not None:
            self.last_sighting = (enemy, self.current_position, step_number)

    def _update_memory_mode(self, enemy):
        # A fresh sighting provides evidence to resume target-memory tracking.
        # Movement alone does not clear the latch and repeatedly restart a wait.
        if enemy is not None:
            self.memory_stall_steps = 0
            self.memory_fallback = False
        else:
            self.memory_stall_steps = (self.memory_stall_steps + 1
                                       if self.current_position == self.memory_observer else 1)
            if (self.memory_stall_limit and self.memory_stall_steps >= self.memory_stall_limit
                    and len(self.ghost_actions[self.current_position]) <= 3):
                self.memory_fallback = True
        self.memory_observer = self.current_position

    @staticmethod
    def _compact_enemy_memories(memories, limit=8):
        if len(memories) <= limit:
            return memories
        ranked = sorted((key for key in memories if key not in ('full', None)),
                        key=lambda key: memories[key].sum(), reverse=True)
        keep = set(ranked[:limit - 2]) | {'full', None}
        result = {key: values for key, values in memories.items() if key in keep}
        for key, values in memories.items():
            if key not in keep:
                result[None] = result.get(None, 0) + values
        return result

    def _search_policy(self):
        belief = self.belief
        if belief['search_policy'] is not None:
            return belief['search_policy']
        edges = belief['edges']
        size, width = edges.shape
        rows = np.arange(size)
        stride = belief['ordered'].shape[1] // len(Move)
        endpoints = belief['ordered'].reshape(size, len(Move), stride)[:, :4, -1]
        moving = endpoints != rows[:, None]
        preference = np.exp(-self.search_exploration * np.minimum(20, belief['enemy_visits'][endpoints])) * moving
        total = preference.sum(axis=1, keepdims=True)
        preference = np.divide(preference, total, out=np.zeros_like(preference), where=total > 0)
        policy = np.zeros_like(edges, dtype=float)
        for direction in range(4):
            slots = (edges == endpoints[:, direction, None]).argmax(axis=1)
            policy[rows, slots] += preference[:, direction]
        stuck = total[:, 0] == 0
        slots = (edges == rows[:, None]).argmax(axis=1)
        policy[rows[stuck], slots[stuck]] = 1
        uniform = (np.arange(width)[None, :] < belief['counts'][:, None]) / belief['counts'][:, None]
        belief['search_policy'] = 0.5 * policy + 0.5 * uniform
        return belief['search_policy']

    def _predict_enemy_memories(self, memories, ghost):
        """Two vision hypotheses: full vision, or remembered targets under fog.

        Target keys represent what the opponent may remember, not its true state.
        None is searching; a small bounded mixture retains alternative sightings.
        """
        return self._predict_enemy_memories_batch([memories], [ghost])[0]

    def _predict_enemy_memories_batch(self, memories_batch, ghosts):
        self._check_time()
        belief = self.belief
        size = len(belief['nodes'])
        labels, probabilities, policies = [], [], []
        policy_cache = {}

        def policy(target):
            if target not in policy_cache:
                policy_cache[target] = self._pursuit_policy(target)
            return policy_cache[target]

        def add(parent, label, distribution, kernel):
            if distribution.sum() > 1e-15:
                labels.append((parent, label))
                probabilities.append(distribution)
                policies.append(kernel)

        for parent, (memories, ghost) in enumerate(zip(memories_batch, ghosts)):
            visible = np.zeros(size, dtype=bool)
            visible[[belief['index'][pos] for pos in self._visible_from(ghost)]] = True
            observed = np.zeros(size)
            for target, distribution in memories.items():
                if target == 'full':
                    add(parent, target, distribution, policy(ghost))
                    continue
                observed += distribution * visible
                hidden = distribution * ~visible
                if target is None:
                    add(parent, None, hidden, self._search_policy())
                else:
                    reached = belief['capture'][belief['index'][target]]
                    add(parent, None, hidden * reached, self._search_policy())
                    add(parent, target, hidden * ~reached, policy(target))
            add(parent, ghost, observed, policy(ghost))
        if not labels:
            return [{} for _ in ghosts]
        endpoints = belief['edges'].ravel()[None, :] + size * np.arange(len(labels))[:, None]
        prediction = np.bincount(endpoints.ravel(),
                                 weights=(np.asarray(probabilities)[:, :, None] * np.asarray(policies)).ravel(),
                                 minlength=len(labels) * size).reshape(len(labels), size)
        result = [{} for _ in ghosts]
        for (parent, label), distribution in zip(labels, prediction):
            result[parent][label] = result[parent].get(label, 0) + distribution
        return [self._compact_enemy_memories(memories) for memories in result]

    def _update_enemy_memories(self, enemy, step_number, learned, streaks):
        belief = self.belief
        size = len(belief['nodes'])
        memories = belief['enemy_memories']
        if memories is None:
            prior = (belief['probabilities'] if belief['probabilities'] is not None
                     else np.full(size, 1 / size))
            memories = {'full': self.memory_full_prior * prior,
                        None: (1 - self.memory_full_prior) * prior}
        if belief['step'] is not None:
            for _ in range(min(size, max(0, step_number - belief['step']))):
                memories = self._predict_enemy_memories(memories, belief['observer'])
        if enemy is not None:
            mask = np.zeros(size, dtype=bool)
            mask[belief['index'][enemy]] = True
        else:
            mask = np.asarray([pos in self.possible_enemies for pos in belief['nodes']])
            mask &= ~belief['capture'][belief['index'][self.current_position]]
        memories = {key: values * mask for key, values in memories.items()}
        mass = sum(values.sum() for values in memories.values())
        if mass <= 1e-12:
            memories = {'full': self.memory_full_prior * mask.astype(float),
                        None: (1 - self.memory_full_prior) * mask.astype(float)}
            mass = sum(values.sum() for values in memories.values())
        if mass <= 0:
            return
        memories = {key: values / mass for key, values in memories.items()}
        for key, weight in (('full', self.memory_full_prior), (None, 1 - self.memory_full_prior)):
            memories[key] = memories.get(key, 0) + 1e-6 * weight * mask
        marginal = sum(memories.values())
        mass = marginal.sum()
        memories = {key: values / mass for key, values in memories.items()}
        self._check_time()
        if learned is not None:
            self.model_weights = learned
            self.model_streaks = streaks
            self.model_observations += 1
            belief['policies'].clear()
        belief['enemy_memories'] = memories
        belief['probabilities'] = marginal / mass
        belief['enemy_visits'] += belief['probabilities']
        belief['search_policy'] = None
        belief['observer'], belief['step'] = self.current_position, step_number
        if enemy is not None:
            self.last_sighting = (enemy, self.current_position, step_number)

    def _pursuit_policy(self, ghost):
        # Trả ma trận [ô Pacman, lựa chọn đích], trộn các kiểu đuổi nhắm vị trí ghost.
        belief = self.belief
        if ghost in belief['policies']:
            belief['policies'].move_to_end(ghost)
            return belief['policies'][ghost]
        models = self._pursuit_models(ghost)
        prior = self.fog_prior if self.fog_enabled and self.model_observations == 0 else self.model_weights
        weights = np.tensordot(prior, models, axes=(0, 0))
        self._check_time()
        belief['policies'][ghost] = weights
        if len(belief['policies']) > belief['cache_limit']:
            belief['policies'].popitem(last=False)
        return weights

    def _pursuit_models(self, ghost):
        # Evaluate all source cells together; padded edges must carry zero mass.
        belief = self.belief
        if ghost in belief['models']:
            belief['models'].move_to_end(ghost)
            return belief['models'][ghost]
        distances = self._distance_map(ghost)
        edges, counts, ordered = belief['edges'], belief['counts'], belief['ordered']
        size, width = edges.shape
        rows = np.arange(size)
        models = np.zeros((len(self.model_prior), size, width), dtype=np.float32)
        distance = np.asarray([distances.get(pos, UNREACHABLE_DISTANCE) for pos in belief['nodes']])
        valid = np.arange(width)[None, :] < counts[:, None]
        costs = np.where(valid, distance[edges], np.inf)
        closest = valid & (costs == costs.min(axis=1, keepdims=True))
        models[MODEL_BFS_RANDOM] = closest / closest.sum(axis=1, keepdims=True)
        models[MODEL_RANDOM] = valid / counts[:, None]
        stride = ordered.shape[1] // len(Move)
        for model, choices in ((MODEL_BFS, ordered),
                               (MODEL_BFS_REVERSE, ordered[:, ::-1]),
                               (MODEL_BFS_SLOW, ordered[:, ::stride])):
            endpoints = choices[rows, distance[choices].argmin(axis=1)]
            slots = (edges == endpoints[:, None]).argmax(axis=1)
            models[model, rows, slots] = 1

        endpoints = self._intercept_targets(ghost, distance)
        slots = (edges == endpoints[:, None]).argmax(axis=1)
        models[MODEL_INTERCEPT, rows, slots] = 1

        directional = ordered.reshape(size, len(Move), stride)
        dr = ghost[0] - belief['coordinates'][:, 0]
        dc = ghost[1] - belief['coordinates'][:, 1]
        vertical = directional[rows, np.where(dr > 0, 1, 0),
                               np.maximum(0, np.minimum(np.abs(dr), stride) - 1)]
        horizontal = directional[rows, np.where(dc > 0, 3, 2),
                                 np.maximum(0, np.minimum(np.abs(dc), stride) - 1)]
        vertical = np.where(dr != 0, vertical, rows)
        horizontal = np.where(dc != 0, horizontal, rows)
        fallback = directional[:, :4, -1]
        moving = fallback != rows[:, None]
        fallback_counts = moving.sum(axis=1)
        for model, first, second in ((MODEL_GREEDY_VERTICAL, vertical, horizontal),
                                     (MODEL_GREEDY_HORIZONTAL, horizontal, vertical)):
            endpoints = np.where(first != rows, first, second)
            chosen = endpoints != rows
            slots = (edges == endpoints[:, None]).argmax(axis=1)
            models[model, rows[chosen], slots[chosen]] = 1
            for direction in range(4):
                active = ~chosen & moving[:, direction]
                slots = (edges == fallback[:, direction, None]).argmax(axis=1)
                models[model, rows[active], slots[active]] += 1 / fallback_counts[active]
            stuck = ~chosen & (fallback_counts == 0)
            slots = (edges == rows[:, None]).argmax(axis=1)
            models[model, rows[stuck], slots[stuck]] = 1
        self._check_time()
        belief['models'][ghost] = models
        if len(belief['models']) > belief['cache_limit']:
            belief['models'].popitem(last=False)
        return models

    def _intercept_targets(self, ghost, distance):
        belief = self.belief
        ordered = belief['ordered']
        ghost_targets = np.asarray([pos for _, pos in self.ghost_actions[ghost]], dtype=np.int32)
        coordinates = belief['coordinates'][ordered]
        differences = (np.abs(coordinates[:, :, None, 0] - ghost_targets[None, None, :, 0])
                       + np.abs(coordinates[:, :, None, 1] - ghost_targets[None, None, :, 1]))
        escape_counts = (differences >= self.capture_distance).sum(axis=2)
        eligible = escape_counts == escape_counts.min(axis=1, keepdims=True)
        worst_distance = differences.max(axis=2)
        best_distance = np.where(eligible, worst_distance, np.inf).min(axis=1, keepdims=True)
        eligible &= worst_distance == best_distance
        maze_cost = distance[ordered]
        best_cost = np.where(eligible, maze_cost, np.inf).min(axis=1, keepdims=True)
        eligible &= maze_cost == best_cost
        return ordered[np.arange(len(ordered)), eligible.argmax(axis=1)]

    def _model_exploitable(self):
        # Trả True khi đủ quan sát và một kiểu greedy chiếm trọng số cao để dự đoán.
        greedy_weight = max(self.model_weights[MODEL_GREEDY_VERTICAL],
                            self.model_weights[MODEL_GREEDY_HORIZONTAL])
        weight = max(greedy_weight, self.model_weights[MODEL_BFS_RANDOM])
        return ((self.model_observations >= MIN_MODEL_OBSERVATIONS
                 and weight >= MODEL_CONFIDENCE_THRESHOLD)
                or (self.model_observations >= 1 and weight >= 0.75))

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
            if len(geometry) > DISTANCE_CACHE_SIZE:
                geometry.popitem(last=False)
        else:
            geometry.move_to_end(pos)
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
        horizon = min(self.forecast_horizon, self.max_depth, remaining)
        memory = self.opponent_memory and self.fog_enabled and belief['enemy_memories'] is not None
        distribution = belief['enemy_memories'] if memory else belief['probabilities']
        states = [(self.current_position, distribution, None)]
        try:
            for depth in range(1, horizon + 1):
                expand = self._memory_candidates if memory else self._forecast_candidates
                candidates = expand(states, allowed if depth == 1 else None)
                candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                selected = self._select_forecast_states(candidates)
                if not selected:
                    break
                best_move = selected[0][1]
                states = [(pos, probabilities, first)
                          for _, first, pos, probabilities in selected]
        except SearchTimeout:
            pass
        return best_move

    def _forecast_batch(self, distributions, ghosts):
        """Propagate a beam with one scatter rather than one scatter per path."""
        self._check_time()
        belief = self.belief
        probabilities = np.asarray(distributions)
        kernels = np.asarray([self._pursuit_policy(ghost) for ghost in ghosts])
        base = belief['edges'].ravel()
        cells_per_path = int(np.prod(probabilities.shape[1:]))
        endpoints = base[None, :] + cells_per_path * np.arange(len(ghosts))[:, None]
        prediction = np.bincount(endpoints.ravel(),
                                 weights=(probabilities[..., None] * kernels).ravel(),
                                 minlength=probabilities.size).reshape(probabilities.shape)
        return prediction

    def _forecast_candidates(self, states, allowed=None):
        predictions = self._forecast_batch([state[1] for state in states],
                                            [state[0] for state in states])
        paths = [(parent, first if first is not None else move, pos)
                 for parent, (ghost, _, first) in enumerate(states)
                 for move, pos in self.ghost_actions[ghost]
                 if allowed is None or move in allowed]
        if not paths:
            return []
        self._check_time()
        positions = [path[2] for path in paths]
        capture = self.belief['capture'][[self.belief['index'][pos] for pos in positions]]
        following = predictions[[path[0] for path in paths]]
        following *= ~capture
        marginals = following
        utilities = self._forecast_utilities(positions, marginals)
        return [(float(utility), path[1], path[2], distribution)
                for utility, path, distribution in zip(utilities, paths, following)]

    def _memory_candidates(self, states, allowed=None):
        paths, distributions, marginals = [], [], []
        predictions = self._predict_enemy_memories_batch([state[1] for state in states],
                                                        [state[0] for state in states])
        for parent, (ghost, memories, first) in enumerate(states):
            predicted = predictions[parent]
            for move, pos in self.ghost_actions[ghost]:
                if allowed is not None and move not in allowed:
                    continue
                self._check_time()
                safe = ~self.belief['capture'][self.belief['index'][pos]]
                following = {key: values * safe for key, values in predicted.items()}
                paths.append((parent, first if first is not None else move, pos))
                distributions.append(following)
                marginals.append(sum(following.values()) if following else
                                 np.zeros(len(self.belief['nodes'])))
        if not paths:
            return []
        utilities = self._forecast_utilities([path[2] for path in paths], np.asarray(marginals))
        return [(float(utility), path[1], path[2], distribution)
                for utility, path, distribution in zip(utilities, paths, distributions)]

    def _forecast_utilities(self, positions, marginals):
        mass = marginals.sum(axis=1)
        geometry = {pos: self._turn_geometry(pos) for pos in dict.fromkeys(positions)}
        distances = np.asarray([geometry[pos][0] for pos in positions])
        turns = np.asarray([geometry[pos][1] for pos in positions])
        space = np.einsum('ij,ij->i', marginals, distances)
        delay = np.einsum('ij,ij->i', marginals, turns)
        information = self._information_gains(positions, marginals, mass)
        bias = np.asarray([0.2 * (geometry[pos][2] + 0.5 * (len(self.ghost_actions[pos]) > 3))
                           + len(self.ghost_actions[pos]) - 1 - 4 * (pos in self.trapped)
                           - 0.1 * self.visits[pos] for pos in positions])
        utilities = 10000 * mass + 0.4 * space + 0.3 * delay + mass * (bias + 0.25 * information)
        self._check_time()
        return utilities

    def _information_gains(self, positions, probabilities, mass):
        if not self.fog_enabled:
            return np.zeros(len(positions))
        belief = self.belief
        masks = belief['visibility_masks']
        rows = {}
        for pos in dict.fromkeys(positions):
            key = (pos, self.vision_radius)
            if key not in masks:
                visible = np.zeros(len(belief['nodes']), dtype=bool)
                visible[[belief['index'][cell] for cell in self._visible_from(pos)]] = True
                masks[key] = visible
                if len(masks) > DISTANCE_CACHE_SIZE:
                    masks.popitem(last=False)
            else:
                masks.move_to_end(key)
            rows[pos] = masks[key]
        visible = probabilities * np.asarray([rows[pos] for pos in positions])
        visible = np.divide(visible, mass[:, None], out=np.zeros_like(visible), where=mass[:, None] > 0)
        outside = np.maximum(0, 1 - visible.sum(axis=1))
        logarithms = np.log2(visible, out=np.zeros_like(visible), where=visible > 0)
        outside_log = np.log2(outside, out=np.zeros_like(outside), where=outside > 0)
        return -np.einsum('ij,ij->i', visible, logarithms) - outside * outside_log

    def _select_forecast_states(self, candidates):
        # Position alone is not a belief state: different paths can leave
        # different enemy distributions and different surviving probability mass.
        unique, seen = [], set()
        for candidate in candidates:
            distribution = candidate[3]
            key = (candidate[1], candidate[2])
            if self.belief_merge == 'exact':
                if isinstance(distribution, dict):
                    key += (tuple(sorted((repr(label), values.tobytes())
                                          for label, values in distribution.items())),)
                else:
                    key += (distribution.shape, distribution.tobytes())
            if key not in seen:
                unique.append(candidate)
                seen.add(key)
        leaders = {}
        for candidate in unique:
            leaders.setdefault(candidate[1], candidate)
        selected = list(leaders.values())
        for candidate in unique:
            if len(selected) >= self.forecast_beam_width:
                break
            if candidate is not leaders[candidate[1]]:
                selected.append(candidate)
        return selected

    def _distance_map(self, start):
        # Trả {ô: số bước ngắn nhất từ start} và lưu cache; bỏ các ô không tới được.
        if start not in self.maze_distance_cache:
            self.maze_distance_cache[start] = self._breadth_first_distances((start,))
            if len(self.maze_distance_cache) > DISTANCE_CACHE_SIZE:
                self.maze_distance_cache.popitem(last=False)
        else:
            self.maze_distance_cache.move_to_end(start)
        return self.maze_distance_cache[start]

    def _expand_enemies(self, enemies):
        # Trả tập đích Pacman có thể tới sau một lượt từ bất kỳ ô nào trong enemies.
        key = frozenset(enemies)
        cached = self.expansion_cache.get(key)
        if cached is not None:
            self.expansion_cache.move_to_end(key)
            return cached
        expanded = set()
        for count, pos in enumerate(enemies):
            if count & 31 == 0:
                self._check_time()
            expanded.update(self.pacman_targets[pos])
        expanded = frozenset(expanded)
        self.expansion_cache[key] = expanded
        if len(self.expansion_cache) > 32:
            self.expansion_cache.popitem(last=False)
        return expanded

    def _breadth_first_distances(self, starts, pacman=False):
        # BFS nhiều nguồn: trả số cạnh ít nhất từ starts tới từng ô có thể tới.
        # pacman=True dùng nước đi tối đa pacman_speed ô làm một cạnh (một lượt).
        key = (frozenset(starts), pacman) if len(starts) > 1 else None
        if key is not None and key in self.multi_distance_cache:
            self.multi_distance_cache.move_to_end(key)
            return self.multi_distance_cache[key]
        distances = dict.fromkeys(starts, 0)
        queue = deque(starts)
        iterations = 0
        while queue:
            if iterations & 31 == 0:
                self._check_time()
            iterations += 1
            pos = queue.popleft()
            neighbors = (self.pacman_targets[pos] if pacman else
                         self.ghost_neighbors[pos])
            for nxt in neighbors:
                if nxt not in distances:
                    distances[nxt] = distances[pos] + 1
                    queue.append(nxt)
        if key is not None:
            self.multi_distance_cache[key] = distances
            if len(self.multi_distance_cache) > 32:
                self.multi_distance_cache.popitem(last=False)
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
        key = (ghost, pacman)
        cached = self.evaluation_cache.get(key)
        if cached is not None:
            self.evaluation_cache.move_to_end(key)
            return cached
        distance = self._distance_map(pacman).get(ghost, UNREACHABLE_DISTANCE)
        pacman_distances = self._pacman_distance_map(pacman)
        value = (self._safety_score(ghost, distance, pacman_distances)
                 + 0.5 * min(self._manhattan(ghost, pacman), 20))
        self.evaluation_cache[key] = value
        if len(self.evaluation_cache) > 8192:
            self.evaluation_cache.popitem(last=False)
        return value

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
