import dotenv
import hydra
from omegaconf import DictConfig

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)


# Load hydra configs and call train method.
@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    # Imports should be nested to optimize hydra tab completion.
    from src.train_ensemble import train
    from src.utils import template_utils

    # Setup utilities
    template_utils.extras(config)

    # Pretty print current configs in a tree
    if config.get("print_config"):
        template_utils.print_config(config, resolve=True)

    # Call train method with configs
    train(config)


if __name__ == "__main__":
    main()
