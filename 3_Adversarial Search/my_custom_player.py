
from sample_players import DataPlayer
from isolation import DebugState



class CustomPlayer(DataPlayer):
    nodes_searched=0
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def my_moves(self,state, depth):
        player_id = 0
        score = len(state.liberties(state.locs[0])) - len(state.liberties(state.locs[1])) + 3*self.movesNextMoves(state)-self.overlappingMoves(state)+self.calcManhattanDistance(state)
    #baseline
        #score = len(state.liberties(state.locs[player_id]))-len(state.liberties(state.locs[player_id+1]))
        return score
    
    def takeOppMove(self,state):
        if state.locs[0] in state.liberties(state.locs[1]):
            return 2
        else:
            return 0
        
    def movesNextMoves(self,state):
        #get this position's next moves
        pos1_liberties = state.liberties(state.locs[0])
        maxNextLib=0
        for lib in pos1_liberties:
            nextPosNumLiberties = len(state.liberties(lib))
            if nextPosNumLiberties>maxNextLib:
                maxNextLib=nextPosNumLiberties
        
        return maxNextLib
    
    def overlappingMoves(self,state):
        #higher score with more overlapping moves (?); would let me take their positions
        pos1_liberties = state.liberties(state.locs[0])
        pos2_liberties = state.liberties(state.locs[1])
        num_overlaps = 0
        for i in range(len(pos1_liberties)):
            for j in range(len(pos2_liberties)):
                if pos1_liberties[i]==pos1_liberties[j]:
                    num_overlaps +=1
                    break
        return num_overlaps
    def calcDistFromCenter(self,state):
        ctr = (5,4)
        #calculate coordinates; autograder doesn't have DebugState
        myloc = state.locs[0]
        pos1=[]
        pos1.append(myloc%13)
        pos1.append(myloc//13)
        #pos1=DebugState.ind2xy(state.locs[0])
        dist_from_ctr = abs(ctr[0] - pos1[0]) + abs(ctr[1] - pos1[1])
        #print(9-dist_from_ctr)
        #9 is the max distance from the center
        return (9-dist_from_ctr)
    
    def calcManhattanDistance(self,state):
        #calculate the distance between the two players
        #print(DebugState.ind2xy(57))
        #calc xy coords; autograder doesn't have Debugstate--this should be refactored, but running out of time!
        myloc = state.locs[0]
        opploc=state.locs[1]
        pos1=[]
        pos2=[]
        pos1.append(myloc%13)
        pos1.append(myloc//13)
        pos2.append(opploc%13)
        pos2.append(opploc//13)
        
        #pos1 = DebugState.ind2xy(state.locs[0])
        #pos2 = DebugState.ind2xy(state.locs[1])
        man_dist = abs(pos1[0] - pos2[0])+abs(pos1[1] - pos2[1])
        #min dist between--want to move toward other player; max dist is 20 squares
        return (20-man_dist)
        
    
    def alpha_beta_search(self, state, depth):
      
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        #self.nodes_searched += len(state.actions())

        for a in state.actions():
            v = self.min_value(state.result(a), alpha, beta, depth-1)
            #print("depth ",depth)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        #print("move ",best_move)
        return best_move
    
    def min_value(self, state, alpha, beta, depth):
        self.nodes_searched+=1
        #print("min value ",depth)
        if state.terminal_test()==True:
            return state.utility(0)
        
        if depth<=0:
            #print("depth ",depth)
            #print("min score ",self.my_moves(state,depth))
            return self.my_moves(state, depth)

        v = float("inf")
        #self.nodes_searched+= len(state.actions())

        for a in state.actions():

            v = min(v, self.max_value(state.result(a), alpha, beta, depth-1))
            if v<=alpha:
                return v
            beta=min(beta,v)

        return v
    def max_value(self, state, alpha, beta, depth):
        self.nodes_searched+=1
        #print("max value depth ",depth)
        if state.terminal_test():
            return state.utility(0)
        
        if depth<=0:
            #print("max score ",self.my_moves(state,depth))
            return self.my_moves(state, depth)
        
        v = float("-inf")
        #self.nodes_searched+=len(state.actions())
        for a in state.actions():
            v= max(v, self.min_value(state.result(a), alpha, beta, depth-1))
            if v>=beta:
                return v
            alpha=max(alpha,v)


        return v


            
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        
        #debug_board = DebugState.from_state(state)
        #print(debug_board.bitboard_string)
        #print(debug_board)
        import random
        import time
        startTime = time.time()
        #print("my move")
        #depth = 2
        #assuming branching factor of 11, can search 10.73 levels deep in 150 s
        depth_limit=4
        time_limit=150
        
        best_move = None
        if state.ply_count <= 1:
            if state.player()==0:
                best_move=57
            else:
                best_move = 58
            
            #self.queue.put(first_move)
            #first_move=57
            #self.queue.put(first_move)
        else:
            #print("here")
            for i in range(1,depth_limit+1):
                currentTime=time.time()
                timeElapsed=currentTime-startTime
                #print("i ",i)
                #print("time ",timeElapsed)
                if timeElapsed>=1:
                    #self.queue.put(best_move)
                    break
                else:
                    best_move=self.alpha_beta_search(state,i)
            #print("move ",best_move)
            #self.queue.put(best_move)
        if best_move==None:
            best_move= random.choice(state.actions())
        #print("total depth ",state.ply_count)
        #print("nodes searched ",self.nodes_searched)
        self.queue.put(best_move)
        
        '''
            best_move = self.alpha_beta_search(state,5)
            if best_move==None:
                self.queue.put(random.choice(state.actions()))
            else:
            #print("best ",best_move)
                self.queue.put(best_move) 
            #for i in range(0,depth):
             #   best_move = self.alpha_beta_search(state,i)
             #   self.queue.put(best_move)
                #print("nodes searched: ",self.nodes_searched)
         '''
 
       
        
