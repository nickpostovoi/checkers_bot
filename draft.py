def get_reward(self, player):
        # define the reward rule values
        capture_reward = 5
        king_reward = 10
        win_reward = 100
        loss_penalty = -100
        draw_penalty = -10
        move_penalty = -0.1

        # initialize reward count
        reward = 0

        # check if the game is over
        if self.is_game_over():
            if self.pieces_player_1 == 0 or not self.get_legal_moves(1):
                # if player 2 wins
                reward = win_reward if player == 2 else loss_penalty
            elif self.pieces_player_2 == 0 or not self.get_legal_moves(2):
                # if player 1 wins
                reward = win_reward if player == 1 else loss_penalty
            else:
                # its a draw
                reward = draw_penalty
        
        else:
            # determine the reward based on the current state
            player_captures = 12 - self.pieces_player_2 if player == 1 else 12 - self.pieces_player_1
            opponent_captures = 12 - self.pieces_player_1 if player == 2 else 12 - self.pieces_player_2

            reward += player_captures * capture_reward - opponent_captures * capture_reward

            # check for kings and assign rewards
            for row in self.state:
                for piece in row:
                    if piece and piece.king:
                        if piece.player == player:
                            reward += king_reward
                        else:
                            reward -= king_reward
            
            # apply a penalty for each move
            reward += move_penalty

        return reward