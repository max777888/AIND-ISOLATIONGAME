"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math
import operator
import numpy
class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    opponent = game.get_opponent(player)

    # obtaining locations
    playerMoves  = game.get_legal_moves(player)
    opponentMoves = game.get_legal_moves(opponent)

    # returning heuristic
    return float(len(playerMoves) - len(opponentMoves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    mid_w , mid_h = game.height // 2 + 1 , game.width // 2 + 1
    center_location = (mid_w , mid_h)

    # getting players location
    player_location  = game.get_player_location(player)

    # checking if player is the center location
    if center_location == player_location:
      # returning heuristic1 with incentive 
      return custom_score(game, player)+100
    else:
      # returning heuristic1 
      return custom_score(game, player)

def proximity(location1, location2):
      '''Function return extra score as function of proximity between two positions.
         Parameters
         ----------
         location1, location2: tuple
           two tuples of integers (i,j) correspond two different positions on the board

         Returns
         ----------
           float
         The heuristic value of 100 for center of the board position and zero otherwise   
      '''
      return abs(location1[0]-location2[0])+abs(location1[1]-location2[1])




def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    opponent = game.get_opponent(player)
    player_location = game.get_player_location(player )
    opponent_location = game.get_player_location(opponent)
    playerMoves = game.get_legal_moves(player )
    opponentMoves = game.get_legal_moves(opponent)
    blank_spaces = game.get_blank_spaces()

    board_size = game.width * game.height

    # size of local area
    localArea = (game.width + game.height)/4

    # condition that corresponds to later stages of the game
    if board_size - len(blank_spaces) > float(0.3 * board_size):
      # filtering out moves that are within local are
      playerMoves = [move for move in playerMoves if proximity(player_location, move)<=localArea]
      opponentMoves = [move for move in opponentMoves if proximity(opponent_location, move)<=localArea]

    return float(len(playerMoves) - len(opponentMoves))





class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score,iterative=True,method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.iterative=iterative
        self.method=method

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            print 'before calling the minimaxi'
            print game.to_string()
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        print 'return get_move'
        print best_move
        return best_move

    def min_value(self,game,depth):
        available_min_moves=game.get_legal_moves()
        
        if (not available_min_moves) :
          return (self.score(game, game.active_player), (-1, -1))

        if depth==0:
            return (self.score(game, game.active_player), available_min_moves[0])

        current_min_score = float("inf")
        current_min_move = available_min_moves[0]
        for min_move in available_min_moves:
          forecast_move = game.forecast_move(min_move)
          child_min_score,child_min_move = self.max_value(forecast_move,depth-1)
          if child_min_score <= current_min_score:
              current_min_move = child_min_move
              current_min_score = child_min_score
        return current_min_score , current_min_move

    def max_value(self,game,depth):
        available_max_moves=game.get_legal_moves()
 
        if (not available_max_moves) :
          return (self.score(game, game.active_player), (-1, -1))

        if depth==0:
            return (self.score(game, game.active_player), available_max_moves[0])

        current_max_score = float("-inf")
        current_max_move = available_max_moves[0]
        for move in available_max_moves:
          forecast_max_move = game.forecast_move(current_max_move)
          #print 'max_value'
          #print current_max_move
          #print forecast_max_move.get_legal_moves()
          #print forecast_max_move.to_string()
          child_max_score,child_max_move = self.min_value(forecast_max_move,depth-1)
          if child_max_score >= current_max_score:
               current_max_move = child_max_move
               current_max_score = child_max_score
        return current_max_score , current_max_move


    def minimax(self, game, depth,maximizing_player=True ):
      
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        available_moves=game.get_legal_moves()

        if (not available_moves) :
          return (self.score(game, game.active_player), (-1, -1))

        if depth==0:
          return (self.score(game, game.active_player), available_moves[0])
        print 'minimax start the shit'
        print available_moves
        current_score = float("-inf")
        for move in available_moves:
          child_score, child_move = self.max_value(game.forecast_move(move), depth)
          print 'score' , child_score , 'move ', child_move
          if child_score >= current_score:
               current_move = move
               current_score = child_score
        return current_move
        

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
    
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
       
        self.time_left = time_left
        bestValue = (-1,-1)
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        try:        
          # The search method call (alpha beta or minimax) should happen in
          # here in order to avoid timeout. The try/except block will
          # automatically catch the exception raised by the search method
          # when the timer gets close to expiring
          # pass

          # initializing variables that control iterative deepening, search algorithm and bestMove
          isID = self.iterative
          method = self.method
         
          available_moves = game.get_legal_moves()
    
          if (not available_moves) :
            return (self.score(game, game.active_player), (-1, -1))
       
          if game.move_count == 0:
            return(int(game.height/2), int(game.width/2))

          if isID:   # block corresponding to iterative deepening
            # dict() to stor alternative solution as final selection method could contribute in to overall efficiency
            iterativeDeepening = dict()
            # starting with smalles depth
            depth = 1
            # infinte loop until algorithm doesn't run out 
            while True:
                #calling appropriate search method
                #if  method == 'minimax':
                #    score, move = self.minimax(game, depth, True)
                #elif method == 'alphabeta':                    
                score, move = self.alphabeta(game, depth, alpha=float("-inf"), beta=float("inf"))

                # adding newly received result into dictionary
                print 'get move 00'
                print move
                iterativeDeepening[move] = score
                if move in iterativeDeepening:
                    if score > iterativeDeepening[move]:
                        iterativeDeepening[move] = score
                else:
                    iterativeDeepening[move] = score
                # getting move simply based best heuristic value. Perhaps,  not the cleverest way... 
                bestValue = max(iterativeDeepening.items(), key=operator.itemgetter(1))[0]
                print 'alfa best value'
                print iterativeDeepening.items()
                print bestValue
                # returning current best option before time is out

                if time_left() < 15:
                    return bestValue

                # incrementally increase search depth
                depth +=1
                return bestValue
          else: #fixed depth search branch
            depth = self.search_depth
            move = (-1, -1)
            # either minmax or alphabeta search method
            #if method == 'minimax':
            #    _, move = self.minimax(game, depth, True)
            #elif method == 'alphabeta':                    
            _, move = self.alphabeta(game, depth, alpha=float("-inf"), beta=float("inf"))
            print 'alfa best value 2'
            print move
            return move

        except SearchTimeout:
          # Handle any actions required at timeout, if necessary
          print 'alfa best value 3'
          print bestValue
          return bestValue
          pass
        # Return the best move from the last completed search iteration
        print 'alfa best value 4'
        print bestValue
        return bestValue
        raise NotImplementedError

    def min_value(self,game,depth):
        available_min_moves=game.get_legal_moves()
        
        if (not available_min_moves) :
          return (self.score(game, game.active_player), (-1, -1))

        if depth==0:
            return (self.score(game, game.active_player), available_min_moves[0])

        current_min_score = float("inf")
        current_min_move = available_min_moves[0]
        for min_move in available_min_moves:
          forecast_move = game.forecast_move(min_move)
          child_min_score,child_min_move = self.max_value(forecast_move,depth-1)
          if child_min_score <= current_min_score:
              current_min_move = child_min_move
              current_min_score = child_min_score

          if current_min_score <= alpha:
            return current_min_score, move

          if current_score < beta:
            current_min_move = move
            beta = current_min_score

        return current_min_score , current_min_move

    def max_value(self,game,depth,alpha,beta):
        available_max_moves=game.get_legal_moves()
 
        if (not available_max_moves) :
          return (self.score(game, game.active_player), (-1, -1))

        if depth==0:
            return (self.score(game, game.active_player), available_max_moves[0])

        current_max_score = float("-inf")
        current_max_move = available_max_moves[0]
        for move in available_max_moves:
          forecast_max_move = game.forecast_move(current_max_move)
          #print 'max_value'
          #print current_max_move
          #print forecast_max_move.get_legal_moves()
          #print forecast_max_move.to_string()
          child_max_score,child_max_move = self.min_value(forecast_max_move,depth-1)
          if child_max_score >= current_max_score:
               current_max_move = child_max_move
               current_max_score = child_max_score
            
          if child_max_score >= beta:
            return current_max_score, move

          if current_score > alpha:
            current_max_move = move
            alpha = current_max_score  


        return current_max_score , current_max_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),maximizing_player=True):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures."""

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        available_moves = game.get_legal_moves()
    
        if (not available_moves) :
          return (self.score(game, game.active_player), (-1, -1))
       
        if (depth==0):
          return (self.score(game, game.active_player), available_moves[0])
       
        current_move = available_moves[0]
        current_score = float("-inf")
        for move in available_moves:
          game_forecast = game.forecast_move(move)
          #child_score, child_move = self.alphabeta(game_forecast, depth-1,alpha, beta,False)
          child_score, child_move = self.max_value(game_forecast, depth-1,alpha, beta)
          if child_score >= current_score:
            current_score = child_score

                if current_score >= beta:
                    return move
                    return current_score, move

                if current_score > alpha:
                    current_move = move
                    alpha = current_score

        else:
            current_score = float("inf")
            for move in available_moves:
                game_forecast = game.forecast_move(move)
                child_score, child_move = self.alphabeta(game_forecast, depth-1,
                                                           alpha, beta,
                                                           True)

                if child_score <= current_score:
                    current_score = child_score

                if current_score <= alpha:
                    return move
                    return current_score, move

                if current_score < beta:
                    current_move = move
                    beta = current_score
        print 'alfa current move end'
        return current_score,current_move