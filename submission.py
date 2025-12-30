from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    """
    h(state, robot_id) = CreditDiff + DeliveryProgress + BatterySafety +
                         PackageAcquisition + CompetitiveAdvantage + FuturePlanning
    """

    my_robot = env.get_robot(robot_id)
    opponent_robot = env.get_robot((robot_id + 1) % 2)

    # Credit difference 
    score = 1000 * (my_robot.credit - opponent_robot.credit)

    # Low battery 
    credit_diff = my_robot.credit - opponent_robot.credit
    battery_diff = my_robot.battery - opponent_robot.battery

    if my_robot.battery <= 2:
        if credit_diff > 0 and opponent_robot.battery <= 2:
            score -= 1000 
        else:
            score -= 5000 
    elif my_robot.battery <= 5:
        if credit_diff > 0 and battery_diff >= 0:
            score -= 200
        else:
            score -= 1000

    # Package handling
    if my_robot.package is not None:
        # Holding a package
        package_reward = 2 * manhattan_distance(my_robot.package.position, my_robot.package.destination)
        dist_to_dest = manhattan_distance(my_robot.position, my_robot.package.destination)

        score += 500 * package_reward
        score -= 100 * dist_to_dest

        if dist_to_dest == 0:
            score += 2000
    else:
        # Not holding package
        active_packages = [p for p in env.packages if p.on_board]

        if active_packages:
            best_package_score = float('-inf')

            for package in active_packages:
                package_reward = 2 * manhattan_distance(package.position, package.destination)
                my_dist = manhattan_distance(my_robot.position, package.position)
                opp_dist = manhattan_distance(opponent_robot.position, package.position)

                pkg_score = 300 * package_reward   
                pkg_score -= 80 * my_dist          

                # Competitive advantage
                if my_dist < opp_dist:
                    pkg_score += 200               # Bonus if closer than opponent

                if my_dist == 0:
                    pkg_score += 1500              # Bonus for immediate pickup

                best_package_score = max(best_package_score, pkg_score)

            score += best_package_score

    # Battery management
    if my_robot.credit > 0 and my_robot.battery < 10:
        dist_to_charge = min(manhattan_distance(my_robot.position, cs.position) for cs in env.charge_stations)

        credit_diff = my_robot.credit - opponent_robot.credit
        battery_diff = my_robot.battery - opponent_robot.battery

        opponent_potential = opponent_robot.battery * 2

        should_charge = False
        charge_urgency = 0

        if credit_diff < 0:
            should_charge = True
            charge_urgency = 2
        elif battery_diff < -5:
            if credit_diff < opponent_potential:
                should_charge = True
                charge_urgency = 2
        elif credit_diff > 30 and my_robot.battery < 5:
            should_charge = True
            charge_urgency = 1 

        if should_charge:
            score -= (50 * charge_urgency) * dist_to_charge
            if dist_to_charge == 0:
                score += (400 * charge_urgency)

    # Competitive awareness
    if opponent_robot.package is not None:
        score -= 300

    # Future packages consideration
    future_packages = [p for p in env.packages if not p.on_board]
    if future_packages and my_robot.package is None:
        best_future_value = 0
        for package in future_packages:
            package_reward = 2 * manhattan_distance(package.position, package.destination)
            dist_to_pkg = manhattan_distance(my_robot.position, package.position)
            future_value = 20 * package_reward - 5 * dist_to_pkg
            best_future_value = max(best_future_value, future_value)
        score += best_future_value

    return score




class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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