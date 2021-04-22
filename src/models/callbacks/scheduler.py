# import pytorch_lightning as pl
# from torch.optim import Optimizer

# # flake8: noqa


# class Scheduler(torch.optim.lr_scheduler.MultiplicativeLR):
#     def __init__(
#         self,
#         optimizer: Optimizer,
#         lr_lambda: list,
#         last_epoch: int = -1,
#     ) -> None:
#         self.lr_lambda = lambda epoch: 0.95
