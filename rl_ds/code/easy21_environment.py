import numpy as np
from base_environment import BaseEnvironment


class Easy21Environment(BaseEnvironment):
    def __init__(self, env_settings=None):
        super().__init__(env_settings)
        self.player_sum = 0
        self.dealer_sum = 0

    def start(self):
        # dealer drawing card
        dealer_card = self._draw(force_black=True)
        if dealer_card['color'] == 'black':
            self.dealer_sum += dealer_card['value']
        else:
            self.dealer_sum -= dealer_card['value']

        # print('Dealer card - {}'.format(dealer_card))

        # player drawing card
        player_card = self._draw(force_black=True)
        if player_card['color'] == 'black':
            self.player_sum += player_card['value']
        else:
            self.player_sum -= player_card['value']

        # print('Player card - {}'.format(player_card))

        next_state = (self.dealer_sum, self.player_sum)
        return next_state

    def step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        if action == 'hit':
            player_card = self._draw()
            if player_card['color'] == 'black':
                self.player_sum += player_card['value']
            else:
                self.player_sum -= player_card['value']

            # print('Player card - {}'.format(player_card))

            bust = self._check_bust(self.player_sum)
            if bust:    # player loose
                reward = -1
                done = True
                next_state = (self.dealer_sum, self.player_sum)
                return reward, next_state, done

            # player's taking next turn
            reward = 0
            done = False
            next_state = (self.dealer_sum, self.player_sum)
            return reward, next_state, done

        if action == 'stick':
            while True:
                # dealer drawing card
                dealer_card = self._draw()
                if dealer_card['color'] == 'black':
                    self.dealer_sum += dealer_card['value']
                else:
                    self.dealer_sum -= dealer_card['value']

                # print('Dealer card - {}'.format(dealer_card))

                bust = self._check_bust(self.dealer_sum)
                if bust:    # player won
                    reward = 1
                    done = True
                    next_state = (self.dealer_sum, self.player_sum)
                    return reward, next_state, done

                # The dealer always sticks on any sum of 17 or greater
                if self.dealer_sum >= 17:
                    break

            if self.player_sum > self.dealer_sum:   # player won
                reward = 1
                done = True
                next_state = (self.dealer_sum, self.player_sum)
                return reward, next_state, done

            if self.player_sum < self.dealer_sum:   # player loose
                reward = -1
                done = True
                next_state = (self.dealer_sum, self.player_sum)
                return reward, next_state, done

            # game is draw
            reward = 0
            done = True
            next_state = (self.dealer_sum, self.player_sum)
            return reward, next_state, done

        # some error occur
        return None

    def end(self):
        """Cleanup done after the environment ends"""
        self.player_sum = 0
        self.dealer_sum = 0

    @staticmethod
    def _draw(force_black=False):
        card_value = np.random.choice(10) + 1
        card_color = 'black' if force_black else np.random.choice(['red', 'black'], p=[0.333333, 1 - 0.333333])
        card = dict(value=card_value, color=card_color)
        return card

    @staticmethod
    def _check_bust(sum):
        if sum > 21 or sum < 1:
            return True
        return False
