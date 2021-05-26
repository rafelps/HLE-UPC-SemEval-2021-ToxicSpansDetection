import os
from argparse import ArgumentParser
from zipfile import ZipFile

# os.environ["TOKENIZERS_PARALLELISM"] = 'true'

from pytorch_lightning import Trainer

from .dataset.toxic_spans_dataset import ToxicDataModule
from .models.multi_depth_distilbert import MultiDepthDistilBertModel
from .utils.metrics import f1_score


def compute_ensemble_predictions(predictions):
    n_models = len(predictions)
    n_sentences = len(predictions[0])
    ensemble_predictions = []
    for i in range(n_sentences):
        counter = {}
        p = set()
        for j in range(n_models):
            for e in predictions[j][i]:
                counter[e] = counter.get(e, 0) + 1
                if counter[e] / n_models >= 0.5:
                    p.add(e)
        ensemble_predictions.append(p)
    return ensemble_predictions


def evaluate():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--name', type=str, help='Name of the model to evaluate', default='best')
    parser.add_argument('--split', type=str, help='Validation, Test or both splits', choices=['val', 'test', 'all'],
                        default='val')
    parser.add_argument('--generate_output', action='store_true', help='Generate output for submisison')
    parser.add_argument('--num_workers', type=int, help='Num workers', default=4)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)

    parser = MultiDepthDistilBertModel.add_model_specific_args(parser)

    args = parser.parse_args()

    data_path = 'data'
    weights_path = os.path.join('weights', args.name)

    checkpoint_names = []
    for file in os.listdir(weights_path):
        if file.endswith('.ckpt'):
            checkpoint_names.append(os.path.join(weights_path, file))

    toxic_data_module = ToxicDataModule(data_path, args.batch_size, args.num_workers)

    if len(checkpoint_names) > 1:
        print(f"Evaluating an ensemble of {len(checkpoint_names)} models:\n")

    original_spans = None
    predicted_spans = []
    for i, checkpoint in enumerate(checkpoint_names):
        print(f"Evaluating model {checkpoint} ...")

        model = MultiDepthDistilBertModel.load_from_checkpoint(checkpoint_path=checkpoint)

        toxic_data_module.setup(stage=args.split)
        test_loader = toxic_data_module.test_dataloader()

        trainer = Trainer.from_argparse_args(args, logger=False)
        trainer.test(model, test_dataloaders=test_loader, verbose=False)

        data = model.predictions
        predicted_spans.append(data['predicted_spans'])
        if i == 0:
            original_spans = data['original_spans']

        f1 = f1_score(data['predicted_spans'], original_spans)
        print(f"Result for model {checkpoint} --> f1-score = {f1:.4f}\n")

    if len(checkpoint_names) > 1:
        predicted_spans_ensemble = compute_ensemble_predictions(predicted_spans)
        f1 = f1_score(predicted_spans_ensemble, original_spans)
        print(f"\nResult for ensemble --> f1-score = {f1:.4f}")
    else:
        predicted_spans_ensemble = predicted_spans[0]

    if args.generate_output:
        zip_file = os.path.join(weights_path, 'output.zip')
        out_name = 'spans-pred.txt'
        out_file = os.path.join(weights_path, out_name)
        with open(out_file, 'w') as f:
            for i, offs in enumerate(predicted_spans_ensemble):
                f.write(f"{str(i)}\t{str(sorted(offs))}\n")
        with ZipFile(zip_file, 'w') as f:
            f.write(out_file, arcname=out_name)
