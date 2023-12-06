from handler import Controller

# PARAMS
episodes = 1000
batch_size = 32

ticker = 'AAPL'
period = 'max'
interval = '1mo'

timestep = 0
window_size = 10

if __name__ == "__main__":
    controller = Controller(ticker, period, interval,
                            timestep, window_size, episodes, batch_size)
    controller.handler()
