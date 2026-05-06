from common.logging import configure_logging
from ml.train import train_model


if __name__ == "__main__":
    configure_logging()
    train_model()