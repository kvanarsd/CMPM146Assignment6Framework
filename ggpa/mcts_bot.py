from __future__ import annotations
import math
from copy import deepcopy
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None):
        self.children = {}
        self.parent = parent
        self.results = []
        self.param = param
    
    # value and visits from section
    def value(self):
        return sum(self.results/len(self.results))
    
    def visits(self):
        return len(self.results)
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        # code from section
        exploredNode = []
        all_possible_actions = state.get_actions()
        for action in all_possible_actions:
            if action in self.children.keys():
                exploredNode.append(action)
            else:
                self.children[action] = nextState
        if len(exploredNode) == len(all_possible_actions):
            Select()
        else:
            Expand(exploredNode, all_possible_actions)
        self.select(state)
        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        return random.choice(state.get_actions())
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent = 0):
        pass


    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
    # code from section
        values = []
        visits = []
        actions = []
        for child in self.children.keys():
            actions.append(child)
            values.append(child.value)
            visits.append(child.visit)
        
        weights = []
        for i in range(values):
            weight = values[i] + self.param*math.sqrt(math.log(self.visits())/visits[i])
            weights.append(weight)

        nextAction = random.choices(actions, weights) [0]
        state.step(nextAction)
        self.children[nextAction].expand(state)
        

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state, available): 
        #build a list of unexploredActions
        toExplore = []
        for action in all:
            if action not in explored:
                toExplore.append(action)
        nextAction = random.choice(toExplore)
        expanded_state = state.copy_undeterministic()
        expanded_state.step(nextAction)
        self.children[nextAction] = TreeNode(self.param, self)
        self.children[nextAction].rollout(expanded_state)


    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        while not state.ended():
            action = random.choice(state.get_actions())
            state.step(action)
        return state.score()
        
    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.results.append(result)
        if self.parent is not None:
            self.parent.backpropagate(result)
        
    # RECOMMENDED: You can start by just using state.score() as the actual value you are 
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state): 
        return state.score()
        
        
# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state)) # fallback option
        return best_action.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
