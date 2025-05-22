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
    def __init__(self, param, parent=None, action=None):
        self.children = {}
        self.parent = parent
        self.results = []
        self.param = param
        self.action = action
    
    # value and visits from section
    def value(self):
        if len(self.results) == 0:
            return 0
        return sum(self.results)/len(self.results)
    
    def visits(self):
        return len(self.results)
    # REQUIRED function
    # Called once per iteration
#     selecting nodes until you reach a leaf node
    # expanding the leaf node
    # performing a complete rollout
    # backpropagating the result
    def step(self, state):
        self.select(state)
        # code from section
        # exploredNode = []
        # all_possible_actions = state.get_actions()
        # for action in all_possible_actions:
        #     if action in self.children.keys():
        #         exploredNode.append(action)
        #     else:
        #         self.children[action] = nextState
        # if len(exploredNode) == len(all_possible_actions):
        #     self.select()
        # else:
        #     self.expand(exploredNode, all_possible_actions)
        # self.select(state)

        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        return random.choice(state.get_actions())
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent = 0):
        space = "   " * indent
        for name, child in self.children.items():
            #print("THIS IS THE CHILD" + child)
            average = "?"
            if (len(child.results) > 0) :
                average = sum(child.results) / len(child.results)

            print(space + name + " " + average)
            child.print_tree(indent + 1)


    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
    # code from section
        # if self.parent != None:
        #     # print("calling select on " + self.action.key())
        #     if self.action.key() == "":
        #         print("end turn")
        #         return
        unexploredNodes = []
        all_possible_actions = state.get_actions()
        if len(all_possible_actions) == 0:
            # print("No possible children")
            return
        # print("Here are our current children ")
        # print(self.children.keys())
        curActions = []
        for action in all_possible_actions:
            curActions.append(action.key())
            # print("From Select this is our hand " + action.key())
            if action.key() not in self.children:
                unexploredNodes.append(action)
        if len(unexploredNodes) > 0:
            # unexploredNode = state.copy_undeterministic()
            # unexploredNode.step(action)
            self.expand(state, unexploredNodes)
            return

        values = []
        visits = []
        actions = []
        for name, child in self.children.items():
            if name in curActions:
                # print("Here are our possible actions " + child.action.key())
                actions.append(child.action)
                values.append(child.value())
                visits.append(child.visits())
        
        weights = []
        for i in range(len(values)):
            weight = 0
            if visits[i] > 0:
                weight = values[i] + self.param*math.sqrt(math.log(self.visits())/visits[i])
            weights.append(weight)

        if sum(weights) > 0:
            nextAction = random.choices(actions, weights) [0]
        else:
            nextAction = random.choice(actions)
        # MAYBE COPY STATE IDK
        state.step(nextAction)
        self.children[nextAction.key()].select(state)

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state, available): 
        nextAction = random.choice(available)
        self.children[nextAction.key()] = TreeNode(self.param, self, nextAction)
        state.step(nextAction)
        self.rollout(state)
        #build a list of unexploredActions
        # toExplore = []
        # for action in all:
        #     if action not in explored:
        #         toExplore.append(action)
        # nextAction = random.choice(toExplore)
        # expanded_state = state.copy_undeterministic()
        # expanded_state.step(nextAction)
        # self.children[nextAction] = TreeNode(self.param, self)
        # self.children[nextAction].rollout(expanded_state)


    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        simState = state.copy_undeterministic()
        while not simState.ended():
            action = random.choice(simState.get_actions())
            simState.step(action)
        self.backpropagate(simState.score())
        
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
