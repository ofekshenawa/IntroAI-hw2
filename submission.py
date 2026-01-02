from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math
from func_timeout import func_timeout, FunctionTimedOut

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    """
    Fast and effective heuristic for minimax/alphabeta.
    """
    me = env.get_robot(robot_id)
    opp = env.get_robot(1 - robot_id)

    # Credit difference is the primary goal
    score = 100 * (me.credit - opp.credit)

    if me.battery <= 0:
        return score - 5000
    if opp.battery <= 0:
        score += 2000

    if me.package:
        # Holding package - deliver it
        pkg = me.package
        reward = 2 * manhattan_distance(pkg.position, pkg.destination)
        dist = manhattan_distance(me.position, pkg.destination)
        score += 50 + 5 * reward - 10 * dist
        if dist == 0:
            score += 200  # Can drop off
    else:
        # Look for packages
        best = -200
        for pkg in env.packages:
            if not pkg.on_board or pkg.position == opp.position:
                continue
            reward = 2 * manhattan_distance(pkg.position, pkg.destination)
            my_dist = manhattan_distance(me.position, pkg.position)
            opp_dist = manhattan_distance(opp.position, pkg.position) if not opp.package else 20
            value = 4 * reward - 15 * my_dist
            if my_dist == 0:
                value += 150  # On package
            elif my_dist < opp_dist:
                value += 40  # Closer
            best = max(best, value)
        score += best

    # Opponent's progress
    if opp.package:
        opp_dist = manhattan_distance(opp.position, opp.package.destination)
        if opp_dist == 0:
            score -= 100
        elif opp_dist == 1:
            score -= 40

    return score

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    class TimeOut(Exception):
        pass

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # Use func_timeout to ensure we never exceed time limit
        try:
            return func_timeout(time_limit * 0.95, self._run_step_internal, args=(env, agent_id, time_limit))
        except FunctionTimedOut:
            operators = env.get_legal_operators(agent_id)
            return operators[0] if operators else None

    def _run_step_internal(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()

        operators = env.get_legal_operators(agent_id)
        if not operators:
            return None
        best_op = operators[0]

        for depth in range(1, 100):
            try:
                time_remaining = time_limit - (time.time() - start_time)
                if time_remaining <= 0.1:
                    break

                iteration_best_op = self.get_op(env, agent_id, depth, start_time, time_limit)
                best_op = iteration_best_op

            except self.TimeOut:
                break

        return best_op

    def get_op(self, env: WarehouseEnv, agent_id, depth, start_time, time_limit):
        if time.time() - start_time > time_limit - 0.1:
            raise self.TimeOut()
        
        operators = env.get_legal_operators(agent_id)
        if not operators:
            return None

        other_robot = (agent_id + 1) % 2

        best_value = -math.inf
        best_immediate = -math.inf  # Tie-breaker: prefer higher immediate heuristic
        best_op = operators[0]

        for op in operators:
            if time.time() - start_time > time_limit - 0.1:
                raise self.TimeOut()
            
            child_env = env.clone()
            child_env.apply_operator(agent_id, op)

            value = self.minimax(child_env, depth - 1, other_robot, agent_id, start_time, time_limit)
            immediate = smart_heuristic(child_env, agent_id)  # For tie-breaking

            # Prefer higher minimax value, break ties with immediate heuristic
            if value > best_value or (value == best_value and immediate > best_immediate):
                best_value = value
                best_immediate = immediate
                best_op = op

        return best_op
    
    def minimax(self, env: WarehouseEnv, depth, current_agent_id, original_agent_id, start_time, time_limit):
        if time.time() - start_time > time_limit - 0.1:
            raise self.TimeOut()
        
        if depth == 0 or env.done():
            return smart_heuristic(env, original_agent_id)
        
        operators = env.get_legal_operators(current_agent_id)
        if not operators:
            return smart_heuristic(env, original_agent_id)

        is_max = (current_agent_id == original_agent_id)
        other_robot = (current_agent_id + 1) % 2

        if is_max:
            value = -math.inf
            for op in operators:
                if time.time() - start_time > time_limit - 0.1:
                    raise self.TimeOut()
                child_env = env.clone()
                child_env.apply_operator(current_agent_id, op)
                value = max(value, self.minimax(child_env, depth - 1, other_robot, original_agent_id, start_time, time_limit))
            return value
        else:
            value = math.inf
            for op in operators:
                if time.time() - start_time > time_limit - 0.1:
                    raise self.TimeOut()
                child_env = env.clone()
                child_env.apply_operator(current_agent_id, op)
                value = min(value, self.minimax(child_env, depth - 1, other_robot, original_agent_id, start_time, time_limit))
            return value


class AgentAlphaBeta(Agent):
    class TimeOut(Exception):
        pass

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        try:
            return func_timeout(time_limit * 0.95, self._run_step_internal, args=(env, agent_id, time_limit))
        except FunctionTimedOut:
            operators = env.get_legal_operators(agent_id)
            return operators[0] if operators else None

    def _run_step_internal(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()

        operators = env.get_legal_operators(agent_id)
        if not operators:
            return None
        best_op = operators[0]

        for depth in range(1, 100):
            time_remaining = time_limit - (time.time() - start_time)
            if time_remaining <= 0.1:
                break
            try:
                iteration_best_op = self.alpha_beta_get_op(env, agent_id, depth, start_time, time_limit, best_op)
                best_op = iteration_best_op

            except self.TimeOut:
                break

        return best_op

    def alpha_beta_get_op(self, env: WarehouseEnv, agent_id, depth, start_time, time_limit, fallback_op):
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining <= 0.05:
            raise self.TimeOut()

        operators = env.get_legal_operators(agent_id)
        if not operators:
            return None
        
        operators.sort(key=lambda op: self.quick_heuristic(op), reverse=True)

        # Check the best operation from last iteration first for better pruning
        if fallback_op in operators:
            operators.remove(fallback_op)
            operators.insert(0, fallback_op)

        other_robot = (agent_id + 1) % 2
        best_value = -math.inf
        best_immediate = -math.inf  # Tie-breaker
        best_op = operators[0]
        alpha = -math.inf
        beta = math.inf

        for op in operators:
            if time.time() - start_time > time_limit - 0.1:
                raise self.TimeOut()

            child_env = env.clone()
            child_env.apply_operator(agent_id, op)

            value = self.alpha_beta_minimax(child_env, depth - 1, other_robot, agent_id, alpha, beta, start_time, time_limit)
            immediate = smart_heuristic(child_env, agent_id)

            # Prefer higher minimax value, break ties with immediate heuristic
            if value > best_value or (value == best_value and immediate > best_immediate):
                best_value = value
                best_immediate = immediate
                best_op = op

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_op

    def alpha_beta_minimax(self, env: WarehouseEnv, depth, current_agent_id, original_agent_id, alpha, beta, start_time, time_limit):
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining <= 0.05:
            raise self.TimeOut()
        
        if depth == 0 or env.done():
            return smart_heuristic(env, original_agent_id)

        operators = env.get_legal_operators(current_agent_id)
        if not operators:
            return smart_heuristic(env, original_agent_id)
        
        is_max = (current_agent_id == original_agent_id)
        other_robot = (current_agent_id + 1) % 2

        if is_max:
            value = -math.inf
            for op in operators:
                if time.time() - start_time > time_limit - 0.1:
                    raise self.TimeOut()
                child_env = env.clone()
                child_env.apply_operator(current_agent_id, op)
                value = max(value, self.alpha_beta_minimax(child_env, depth - 1, other_robot, original_agent_id, alpha, beta, start_time, time_limit))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        else:
            value = math.inf
            for op in operators:
                if time.time() - start_time > time_limit - 0.1:
                    raise self.TimeOut()
                child_env = env.clone()
                child_env.apply_operator(current_agent_id, op)
                value = min(value, self.alpha_beta_minimax(child_env, depth - 1, other_robot, original_agent_id, alpha, beta, start_time, time_limit))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value
    
    # TODO: improve this heuristic, keep it fast
    def quick_heuristic(self, op):
        if op == 'drop off': return 100
        if op == 'pick up': return 50
        if 'move' in op: return 1
        return 0


class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        best_op = None
        start_time = time.time()

        for depth in range(1, 100):
            try:
                time_remaining = time_limit - (time.time() - start_time)
                safe_time = time_remaining - 0.05
                if safe_time <= 0:
                    break

                iteration_best_op = func_timeout(safe_time, self.get_op, args=(env, agent_id, depth))

                best_op = iteration_best_op

            except FunctionTimedOut:
                break

        return best_op

    def get_op(self, env: WarehouseEnv, agent_id, depth):
        operators = env.get_legal_operators(agent_id)
        if not operators:
            return None

        other_robot = (agent_id + 1) % 2

        best_value = -math.inf
        best_immediate = -math.inf  # Tie-breaker
        best_op = operators[0]

        for op in operators:
            child_env = env.clone()
            child_env.apply_operator(agent_id, op)

            value = self.expectimax(child_env, depth - 1, other_robot, agent_id)
            immediate = smart_heuristic(child_env, agent_id)

            # Prefer higher expectimax value, break ties with immediate heuristic
            if value > best_value or (value == best_value and immediate > best_immediate):
                best_value = value
                best_immediate = immediate
                best_op = op

        return best_op

    def expectimax(self, env: WarehouseEnv, depth, current_agent_id, original_agent_id):
        if depth == 0 or env.done():
            return smart_heuristic(env, original_agent_id)

        operators = env.get_legal_operators(current_agent_id)
        if not operators:
            return smart_heuristic(env, original_agent_id)

        is_max = (current_agent_id == original_agent_id)
        other_robot = (current_agent_id + 1) % 2

        if is_max:
            value = -math.inf
            for op in operators:
                child_env = env.clone()
                child_env.apply_operator(current_agent_id, op)
                value = max(value, self.expectimax(child_env, depth - 1, other_robot, original_agent_id))
            return value
        else:
            total_weight = 0
            for op in operators:
                if op == "move west" or op == "pick up":
                    total_weight += 3
                else:
                    total_weight += 1

            expected_value = 0
            for op in operators:
                child_env = env.clone()
                child_env.apply_operator(current_agent_id, op)
                child_value = self.expectimax(child_env, depth - 1, other_robot, original_agent_id)

                if op == "move west" or op == "pick up":
                    probability = 3 / total_weight
                else:
                    probability = 1 / total_weight

                expected_value += probability * child_value

            return expected_value


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)