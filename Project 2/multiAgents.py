# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        minFoodDist = None
        minGhostDist = None
        minScaredTime = None
        if newPos in currentGameState.getFood().asList(): minFoodDist = 0
        else:
            for x,y in newFood.asList():
                temp = manhattanDistance(newPos, (x,y))
                if minFoodDist == None or temp < minFoodDist: minFoodDist = temp
        for x,y in newGhostPositions:
            temp = manhattanDistance(newPos, (x,y))
            if minGhostDist == None or temp < minGhostDist: minGhostDist = temp
        for x in newScaredTimes:
            if minScaredTime == None or x < minScaredTime: minScaredTime = x
        if minGhostDist == 0: minGhostDist -= 1
        if minScaredTime > 0: score = minFoodDist
        else: score = minGhostDist - (minFoodDist * 1.1)
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        act = self.findBest(gameState, 0, 0)[1]
        return act
    
    def findBest(self, state, agent, depth):
        if state.isLose() or state.isWin() or depth == self.depth:
            return (self.evaluationFunction(state), None)
        if agent == 0:
            maxScore = None
            maxAction = None
            for action in state.getLegalActions(0):
                tscore = self.findBest(state.generateSuccessor(agent,action), 1, depth)[0]
                if maxScore == None or tscore > maxScore: 
                    maxAction = action
                    maxScore = tscore
            return (maxScore, maxAction)
        else: 
            nAgent = agent + 1
            if state.getNumAgents() == nAgent: nAgent = 0
            if nAgent == 0: depth += 1
            minScore = None
            for action in state.getLegalActions(agent):
                tscore = self.findBest(state.generateSuccessor(agent, action), nAgent, depth)[0]
                if minScore == None or tscore < minScore:
                    minScore = tscore
            return (minScore, None)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.findBest(gameState, 0, 0, float('-inf'), float('inf'))[1]

    def findBest(self, state, agent, depth, alpha, beta):
        if state.isLose() or state.isWin() or depth == self.depth:
            return (self.evaluationFunction(state), None)
        if agent == 0:
            maxScore = None
            maxAction = None
            for action in state.getLegalActions(0):
                tscore = self.findBest(state.generateSuccessor(agent,action), 1, depth, alpha, beta)[0]
                if maxScore == None or tscore > maxScore: 
                    maxAction = action
                    maxScore = tscore
                alpha = max(alpha, tscore)
                if alpha > beta: return (alpha, None)
            return (maxScore, maxAction)
        else: 
            nAgent = agent + 1
            if state.getNumAgents() == nAgent: nAgent = 0
            if nAgent == 0: depth += 1
            minScore = None
            for action in state.getLegalActions(agent):
                tscore = self.findBest(state.generateSuccessor(agent, action), nAgent, depth, alpha, beta)[0]
                if minScore == None or tscore < minScore:
                    minScore = tscore
                beta = min(beta, tscore)
                if alpha > beta: return (beta, None)
            return (minScore, None)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.findBest(gameState, 0, 0)[1]
    
    def findBest(self, state, agent, depth):
        if state.isLose() or state.isWin() or depth == self.depth:
            return (self.evaluationFunction(state), None)
        if agent == 0:
            maxScore = None
            maxAction = None
            for action in state.getLegalActions(0):
                tscore = self.findBest(state.generateSuccessor(agent,action), 1, depth)[0]
                if maxScore == None or tscore > maxScore: 
                    maxAction = action
                    maxScore = tscore
            return (maxScore, maxAction)
        else: 
            nAgent = agent + 1
            if state.getNumAgents() == nAgent: nAgent = 0
            if nAgent == 0: depth += 1
            expScore = 0
            cscore = 0
            for action in state.getLegalActions(agent):
                tscore = self.findBest(state.generateSuccessor(agent, action), nAgent, depth)[0]
                expScore += tscore
                cscore += 1
            avgscore = expScore / cscore
            return (avgscore, None)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance
    pos = currentGameState.getPacmanPosition()
    bks = currentGameState.getGhostPositions() #BK = Bad Kid
    foodList = currentGameState.getFood().asList()
    capList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    numFood = currentGameState.getNumFood()
    minFood = 100
    minGhost = 100
    scared = True
    for x,y in foodList:
        temp = manhattanDistance(pos, (x,y))
        if minFood == None or temp < minFood: minFood = temp
    if all(x == 0 for x in scaredTimes): # if none of the ghosts are scared, consider capsules as food
        scared = False
        '''
        for x,y in capList:
            temp = manhattanDistance(pos, (x,y))
            if minFood == None or temp < minFood: minFood = temp
        '''
    for x,y in bks:
        temp = manhattanDistance(pos, (x,y))
        if minGhost == None or temp < minGhost: minGhost = temp
    if scared == True: return -abs(numFood + minFood) + currentGameState.getScore()
    else: return minGhost - (numFood*1.1) - (minFood*2) + currentGameState.getScore() 

# Abbreviation
better = betterEvaluationFunction
