import os

# os.environ["TOKENIZERS_PARALLELISM"] = 'true'

from argparse import ArgumentParser

from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from .models.multi_depth_distilbert import MultiDepthDistilBertModel
from .dataset.toxic_spans_dataset import ToxicDataModule


def train():
    # Parse args
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--job_name', type=str, help='Name of the job', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--num_workers', type=int, help='Num workers', default=4)

    parser = MultiDepthDistilBertModel.add_model_specific_args(parser)

    args = parser.parse_args()

    # Load data and models
    data_path = 'data'
    toxic_data_module = ToxicDataModule(data_path, args.batch_size, args.num_workers)
    model = MultiDepthDistilBertModel(args=args)

    # Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs', name=args.job_name)

    # Callbacks
    callbacks = []

    # Save best model checkpoints callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=os.path.join('logs', args.job_name, 'version_' + str(tb_logger.version)),
        filename='{epoch:02d}-{val_f1:.2f}',
        save_top_k=3,
        mode='max',
        save_weights_only=True,
        save_last=False)

    callbacks.append(checkpoint_callback)

    # Train
    trainer = Trainer.from_argparse_args(args, logger=tb_logger, callbacks=callbacks)
    trainer.fit(model, datamodule=toxic_data_module)
