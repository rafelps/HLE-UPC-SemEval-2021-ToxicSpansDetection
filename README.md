# HLE-UPC at SemEval-2021 Task 5: Toxic Spans Detection

Official code for our paper titled [**HLE-UPC at SemEval-2021 Task 5: Multi-Depth DistilBERT for Toxic 
Spans Detection**][paper_url], accepted into the ACL-IJCNLP 2021 Workshop, ACL Proceedings and invited for a Poster Presentation.

## Requirements
This project has been build using:
- [Python][py] 3.7
- [PyTorch][pytorch] 1.7.1
- [PyTorch Lightning][pl] 1.3.1
- [HuggingFace Transformers][hugg] 4.1.1

## Results
The original results, submitted in the **SemEval-2021 competition** were:
- **0.6822 f1-score** in the test set for a single model
- **0.6854 f1-score** in the test set for an ensemble of three models

These metrics pushed us up to the **14th position** in the challenge.

Results reported in this repository, however, are slightly different from the ones reported in the paper due to a minute improvement in our data cleaning process. In this second version, instead of following the guidelines specified in our [paper][paper_url], we applied the data cleaning steps in the following order:

&nbsp;&nbsp;&nbsp;1.&nbsp;&nbsp;Whitespace Trimming  
&nbsp;&nbsp;&nbsp;3.&nbsp;&nbsp;Left-Right-Expansion  
&nbsp;&nbsp;&nbsp;2.&nbsp;&nbsp;Singleton Removal  

Moreover, in this second version, we have taken advantage of the publication of the ground truth for the test set to better assess our model against more data. 

During this stage, we found that the trial set (used as validation), and the test set have significantly different 
data distributions. In fact, our best model in the trial set has (almost) the worst performance in the test set and vice versa.

For this reason, we have also sought models that obtain the best performance overall (over the different sets).


<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <th colspan="2">Model Configuration</th>
    <th rowspan="2">f1-score <br> (trial set)</th>
    <th rowspan="2">f1-score <br> (test set)</th>
  </tr>
  <tr>
    <th>Concatenate last N layers</th>
    <th>Training Epochs</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Best Validation</td>
    <td align="center">4</td>
    <td align="center">3</td>
    <td align="center">0.6978</td>
    <td align="center">0.6787</td>
  </tr>
  <tr>
    <td>Best Test</td>
    <td align="center">3</td>
    <td align="center">2</td>
    <td align="center">0.6725</td>
    <td align="center">0.6842</td>
  </tr>
  <tr>
    <td>Best Overall</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">0.6898</td>
    <td align="center">0.6821</td>
  </tr>
</tbody>
</table>

We provide the weights for the `Best Overall` model, which can be downloaded in the following [link][weights_url] and 
placed at `weights/best/best.ckpt` for evaluation or tagging functions.

At the same time, as stated in the [paper][paper_url], it is possible to use an ensemble of models to futher push the 
performance. We have been able to obtain **0.7005 f1-score in the trial set** and **0.6882 in the test set** with 
different ensembles. However, we present an ensemble that obtains strong performance in both sets:

<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <th >Model Configuration</th>
    <th rowspan="2">f1-score <br> (trial set)</th>
    <th rowspan="2">f1-score <br> (test set)</th>
  </tr>
  <tr>
    <th>Concatenate last N layers</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Model 1</td>
    <td align="center">2</td>
    <td align="center">0.6824</td>
    <td align="center">0.6838</td>
  </tr>
  <tr>
    <td>Model 2</td>
    <td align="center">4</td>
    <td align="center">0.6898</td>
    <td align="center">0.6821</td>
  </tr>
  <tr>
    <td>Model 3</td>
    <td align="center">6</td>
    <td align="center">0.6799</td>
    <td align="center">0.6867</td>
  </tr>
  <tr>
    <td><b>Ensemble</b></td>
    <td align="center"><b>-</b></td>
    <td align="center"><b>0.6912</b></td>
    <td align="center"><b>0.6867</b></td>
  </tr>
</tbody>
</table>

We provide the weights for the models constituting this ensemble in the following [link][weights_url]. These have to 
be placed at `weights/ensemble/*.ckpt` for evaluation and tagging functions.

## Usage
### Train
The trainer is implemented using [PyTorch Lightning][pl], so it accepts any of its `Trainer` class arguments shown in 
its [documentation][pl_trainer].

Apart from those, we have added some additional arguments:
- `--job_name JOB_NAME`. Name for the job. A subdirectory will be created using this name in which logs and checkpoints 
  will be stored.
- `--batch_size BATCH_SIZE`. Default: 8. 
- `--lr LR`. Learning rate. Default: 1e-5.
- `--dropout DROPOUT`. Dropout rate for fully connected layers. Default: 0.25.
- `--label_smoothing LABEL_SMOOTHING`. Smoothing amount for Label Smoothing Loss. Default: 0.1.
- `--concat_last_n CONCAT_LAST_N`. Number of transformer outputs to concatenate before applying the 
  classification layer. Default: 4.
  
The complete usage can be seen by typing:
```
$ python train.py -h
```

For example:
```
$ python train.py --gpus 1 --max_epochs 5 --job_name distilbert_concat_4 --concat_last_n 4 
```
### Evaluate
This script evaluates the performance of a model (or an ensemble) in a particular set of data, and can also be 
used to generate a submission zip file containing the spans considered toxic by the model(s).

Its main arguments are:
- `--name NAME`. Name for the model to evaluate. The script will look for any checkpoint (.ckpt file) in the directory 
  `weights/NAME/`. If there is more than one checkpoint, it will evaluate an ensemble of them.
- `--split SPLIT`. Name of the data split to use for evaluation. Use `val` for the trial set, `test` for the test 
  set, and `all` to use both sets jointly.
- `--generate_output`. Flag to activate the generation of the submission file.

Additionally, arguments for the PyTorch Lightning `Trainer` class can be used.

For example:
```
$ python evaluate.py --gpus 1 --name ensemble --split test
```

### Tag
The tagger can be used to tag the toxicity of a text interactively. It has only two arguments:
- `text`: Text to tag.
- `--name NAME`. Name for the model to use. The script will look for any checkpoint (.ckpt file) in the directory   
  `weights/NAME/`. If there is more than one checkpoint, it will use an ensemble of them.

For example:

```
$ python tag.py "Text to tag" --name best
```


## Citation
You can cite our work using:
```bibtex
@inproceedings{palliser-sans-rial-farras-2021-hle,
    title = "{HLE}-{UPC} at {S}em{E}val-2021 Task 5: Multi-Depth {D}istil{BERT} for Toxic Spans Detection",
    author = "Palliser-Sans, Rafel  and
      Rial-Farr{\`a}s, Albert",
    booktitle = "Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.semeval-1.131",
    doi = "10.18653/v1/2021.semeval-1.131",
    pages = "960--966",
    abstract = "This paper presents our submission to SemEval-2021 Task 5: Toxic Spans Detection. The purpose of this task is to detect the spans that make a text toxic, which is a complex labour for several reasons. Firstly, because of the intrinsic subjectivity of toxicity, and secondly, due to toxicity not always coming from single words like insults or offends, but sometimes from whole expressions formed by words that may not be toxic individually. Following this idea of focusing on both single words and multi-word expressions, we study the impact of using a multi-depth DistilBERT model, which uses embeddings from different layers to estimate the final per-token toxicity. Our quantitative results show that using information from multiple depths boosts the performance of the model. Finally, we also analyze our best model qualitatively.",
}
```


[paper_url]: https://aclanthology.org/2021.semeval-1.131/
[weights_url]: https://www.dropbox.com/sh/92xtl13paxmwk1a/AACqeiYbhq7xCgHTOYjVYITca?dl=0
[pytorch]: https://pytorch.org/
[pl]: https://www.pytorchlightning.ai/
[hugg]: https://huggingface.co/transformers/
[py]: https://www.python.org/
[pl_trainer]: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
