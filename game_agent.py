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
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    # maximizing opponent moves
    return float(len(own_moves) - 1.5*len(opp_moves) )


def custom_score_2(game, player ):
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
      return custom_score_3(game, player)+100
    else:
      #returning heuristic1 
      return custom_score_3(game, player)

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

    return float(  (len(playerMoves)* len(playerMoves)) 
             - 1.5 * (len(opponentMoves)*len(opponentMoves)))





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
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

###############################################################################
class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
### *********************************************************************** ###
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
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        # Return the best move from the last completed search iteration
        
        return best_move
    
### *********************************************************************** ###
    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        """
        Calling the helper function
        evaluate_minimax(self, game, depth, maximize_layer=True) to do recursive
        minmax search.
        """    
        return self.calculate_minimax(game, depth)[1]
    
### *********************************************************************** ###    
    def calculate_minimax(self, game, depth, maximize_layer=True):
        """This is a helper function to  implement minimax(self, game, depth) 
        search function

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximize : boolean
            True if it's a maximize node. False otherwise.

        Returns
        -------
        tuple
            (score, move)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Get legal moves using isolation.Board.get_legal_moves()
        moves_legal = game.get_legal_moves()

        # If there are no legal moves, return (-1,-1)
        if not moves_legal:
            return game.utility(self), (-1, -1)   
        
        # If depth reaches to 0, calculate the heuristic value of the game state 
        if depth == 0:
            return self.score(game, self), (-1, -1)
        
        best_move = None
        
        # Implementing full recursive search procedure for minmax search 
        if maximize_layer:
            # Initilaize best_score to be minimum (worst case)
            best_score = float("-inf")
            for move in moves_legal:
                next_state = game.forecast_move(move)
                # Forecast_move switches the active layer
                score, _ = self.calculate_minimax(next_state, depth - 1, False)
                if score > best_score:
                    best_score, best_move = score, move
        else:
            # Initialize best_score to be maximum (worst case)
            best_score = float("inf")
            for move in moves_legal:
                next_state = game.forecast_move(move)
                # Forecast_move switches the active layer
                score, _ = self.calculate_minimax(next_state, depth - 1, True)
                if score < best_score:
                    best_score, best_move = score, move
        return best_score, best_move
        
        

###############################################################################
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
### *********************************************************************** ###
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
        self.time_left = time_left
       
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        search_depth = 1
        while True:
            try:
                best_move = self.alphabeta(game, search_depth)
                search_depth = search_depth + 1
            except SearchTimeout:
                break
        # Return the best move from the last completed search iteration
        return best_move
         
               
### *********************************************************************** ###
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self.calculate_alphabeta(game, depth, alpha, beta)[1]
    
    def calculate_alphabeta(self, game, depth, alpha, beta, maximize_layer=True):
        """This is a helper function to  implement 
        alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"))

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        tuple
            (score, move)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return game.utility(self), (-1, -1)   
        
        if depth == 0:
            return self.score(game, self), (-1, -1)
        
        best_move = None
        
        if maximize_layer:
           # Initilaize best_score to be minimum (worst case)
            best_score = float("-inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                score, _ = self.calculate_alphabeta(
                    next_state, depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score, best_move = score, move
                if best_score >= beta:
                    break
                alpha = max(alpha, best_score)
        else:
            best_score = float("inf")
            for move  in legal_moves:
                next_state = game.forecast_move(move)
                score, _ = self.calculate_alphabeta(
                    next_state, depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score, best_move = score, move
                if best_score <= alpha:
                    break
                beta = min(beta, best_score)
        return best_score, best_move