from tqdm import tqdm
from ai_trader import AI_Trader
from utils import state_creator, stocker_market, stocks_price_format


class Controller:
    def __init__(self, ticker, period, interval, timestep, window_size, episodes, batch_size):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.timestep = timestep
        self.window_size = window_size
        self.episodes = episodes
        self.batch_size = batch_size

    def setup(self):
        data = stocker_market(ticker=self.ticker,
                              period=self.period, interval=self.interval)
        state, _ = state_creator(data, self.timestep, 5)

        trader = AI_Trader(self.window_size)
        trader.model.summary()

        return data, state, trader

    def handler(self):
        data, state, trader = self.setup()

        for episode in range(self.episodes + 1):
            print("Episode: {}/{}".format(episode, self.episodes))
            state = state_creator(data, 0, self.window_size + 1)

            total_profit = 0
            trader.inventory = []

            for t in tqdm(range(len(data) - 1)):
                action = trader.trade(state)
                next_state = state_creator(data, t + 1, self.window_size + 1)

                reward = 0

                if action == 1:  # buying stocker
                    trader.inventory.append(data[t])
                    print("AI Trader bought: ", stocks_price_format(data[t]))
                elif action == 2 and len(trader.inventory) > 0:  # selling stocker
                    buy_price = trader.inventory.pop(0)

                    reward = max(data[t] - buy_price, 0)
                    total_profit += data[t] - buy_price
                    print("AI Trader sold: ", stocks_price_format(
                        data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))

                if t == len(data) - 2:
                    done = True
                else:
                    done = False

                trader.memory.append((state, action, reward, next_state, done))

                state = next_state

                if done:
                    print("##########################")
                    print("Total Profit: {}".format(total_profit))
                    print("##########################")
                if len(trader.memory) > self.batch_size:
                    trader.batch_train(self.batch_size)

            if episode % 10 == 0:
                trader.model.save("ai_trader_{}.h5".format(episode))
