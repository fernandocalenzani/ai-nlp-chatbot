from handler import Controller


if __name__ == "__main__":
    controller = Controller(ticker, period, interval,
                            timestep, window_size, episodes, batch_size)
    controller.handler()
