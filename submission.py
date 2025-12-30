from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math
from func_timeout import func_timeout, FunctionTimedOut

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    """
    Simplified heuristic with clear priorities:
    1. Credit difference is the ultimate goal
    2. Having a package and being close to destination is good
    3. Being close to available packages is good
    4. Battery management when needed

    KEY INSIGHT: The heuristic evaluates the STATE AFTER an action.
    - After 'pick up': robot has package, dist_to_dest > 0 usually
    - After 'drop off': robot has no package, credit increased
    - After 'move': position changed
    """
    my_robot = env.get_robot(robot_id)
    opponent_robot = env.get_robot((robot_id + 1) % 2)

    # Base score from credit difference (this is the ultimate goal)
    credit_diff = my_robot.credit - opponent_robot.credit
    score = 1000 * credit_diff

    # Get active packages for later use
    active_packages = [p for p in env.packages if p.on_board]

    # PACKAGE HANDLING - most important tactical consideration
    if my_robot.package is not None:
        # Robot is holding a package - deliver it!
        package = my_robot.package
        dest = package.destination
        dist_to_dest = manhattan_distance(my_robot.position, dest)

        # CRITICAL FIX for minimax oscillation:
        # The holding score must ALWAYS be strictly greater than on-package score.
        # Otherwise, minimax looks ahead and prefers the "on package" state over
        # the "holding package" state, causing it to not pick up!
        #
        # Solution: Use a HIGH base for holding that always beats on-package.
        #
        # Holding: 2000 - 50 * dist_to_dest (range: 1500-2000 for dist 0-10)
        # On-package: 500 + 50 * reward (max ~900 for reward=8)
        #
        # Credit gain: 1000 * reward (min 2000 for reward=2)
        # At destination: holding=2000, credit=2000+ → drop-off slightly better

        score += 2000 - 50 * dist_to_dest

    else:
        # Robot is not holding a package - look for packages to pick up
        if active_packages:
            # Find the best package to pursue
            best_package_value = float('-inf')

            for package in active_packages:
                # Skip packages at opponent's position (blocked)
                if package.position == opponent_robot.position:
                    continue

                package_reward = 2 * manhattan_distance(package.position, package.destination)
                my_dist = manhattan_distance(my_robot.position, package.position)
                opp_dist = manhattan_distance(opponent_robot.position, package.position)

                # Value = potential reward - cost to get there
                # Use smaller coefficients to ensure on-package < holding
                pkg_value = 50 * package_reward  # Reduced coefficient
                pkg_value -= 100 * my_dist  # Distance penalty

                # ON the package - modest bonus
                # Must be < minimum holding score (2000 - 50*10 = 1500)
                # So total on-package value should be < 1500
                #
                # On-package = 50*reward - 0 + 500 = 500 + 50*reward
                # For reward=8: 500 + 400 = 900 < 1500 ✓
                # For reward=16: 500 + 800 = 1300 < 1500 ✓
                if my_dist == 0:
                    pkg_value += 500  # On-package bonus
                # Bonus if we're closer than opponent
                elif my_dist < opp_dist:
                    pkg_value += 100

                best_package_value = max(best_package_value, pkg_value)

            if best_package_value != float('-inf'):
                score += best_package_value

        # Future packages (lower priority)
        future_packages = [p for p in env.packages if not p.on_board]
        if future_packages:
            best_future = 0
            for package in future_packages:
                pkg_reward = 2 * manhattan_distance(package.position, package.destination)
                dist = manhattan_distance(my_robot.position, package.position)
                value = 10 * pkg_reward - 20 * dist
                best_future = max(best_future, value)
            score += best_future

    # BATTERY MANAGEMENT - must handle battery before anything else when critical
    if env.charge_stations:
        min_charge_dist = min(manhattan_distance(my_robot.position, cs.position)
                              for cs in env.charge_stations)
    else:
        min_charge_dist = 100  # No charge stations

    if my_robot.battery <= 2:
        # Critical battery - must charge or die
        # Huge penalty that overrides everything except being on charge station
        score -= 5000
        # Strong incentive to get to charge station
        if min_charge_dist == 0:
            # On charge station - very good
            score += 6000  # Overcomes the -5000 penalty
        else:
            score -= 500 * min_charge_dist  # Get to charge station ASAP
    elif my_robot.battery <= 5:
        # Low battery - consider charging
        score -= 300
        # Bonus for being close to charge station
        score -= 50 * min_charge_dist

    # Competitive awareness - small penalty if opponent has package
    if opponent_robot.package is not None:
        score -= 100

    # ANTI-OSCILLATION: When no accessible packages, have a clear goal
    # Check if there are any accessible active packages
    accessible_packages = [p for p in active_packages if p.position != opponent_robot.position]

    if my_robot.package is None and not accessible_packages:
        # No packages to pursue - the robot needs a goal to prevent oscillation
        future_packages = [p for p in env.packages if not p.on_board]

        # Check if on a future package spawn location
        on_future_pkg = any(p.position == my_robot.position for p in future_packages)

        if env.charge_stations:
            min_charge_dist = min(manhattan_distance(my_robot.position, cs.position)
                                  for cs in env.charge_stations)
        else:
            min_charge_dist = 100

        if on_future_pkg:
            # On a future package spawn - strong bonus to stay here and wait
            score += 15000
        elif min_charge_dist == 0 and my_robot.credit > 0:
            # On charge station with credit - strong bonus to stay/charge
            # Need to overcome the credit loss from charging
            score += 10000 + my_robot.credit * 1000
        elif my_robot.credit > 0:
            # Have credit - move toward charge station
            score -= 500 * min_charge_dist
        elif future_packages:
            # No credit - move toward future packages
            best_future_dist = min(manhattan_distance(my_robot.position, p.position)
                                   for p in future_packages)
            score -= 300 * best_future_dist
        else:
            # No future packages - stay near charge station
            score -= 500 * min_charge_dist

    return score

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
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
        best_immediate = -math.inf  # Tie-breaker: prefer higher immediate heuristic
        best_op = operators[0]

        for op in operators:
            child_env = env.clone()
            child_env.apply_operator(agent_id, op)

            value = self.minimax(child_env, depth - 1, other_robot, agent_id)
            immediate = smart_heuristic(child_env, agent_id)  # For tie-breaking

            # Prefer higher minimax value, break ties with immediate heuristic
            if value > best_value or (value == best_value and immediate > best_immediate):
                best_value = value
                best_immediate = immediate
                best_op = op

        return best_op
    
    def minimax(self, env: WarehouseEnv, depth, current_agent_id, original_agent_id):
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
                value = max(value, self.minimax(child_env, depth - 1, other_robot, original_agent_id))
            return value
        else:
            value = math.inf
            for op in operators:
                child_env = env.clone()
                child_env.apply_operator(current_agent_id, op)
                value = min(value, self.minimax(child_env, depth - 1, other_robot, original_agent_id))
            return value


class AgentAlphaBeta(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        best_op = None
        start_time = time.time()

        # TODO: Maybe a waste of time?
        operators = env.get_legal_operators(agent_id)
        if not operators:
            return None
        best_op = operators[0]
        
        for depth in range(1, 100):
            time_remaining = time_limit - (time.time() - start_time)
            if time_remaining <= 0.05:
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
                # TODO: Add time check here too?
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
                # TODO: Add time check here too?
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

    class TimeOut(Exception):
        pass


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