Place your trained weights here inside subdirectories. If a subdirectory contains more than one weights file (.ckpt), the 
evaluator and tagger scripts will interpret them as an ensemble of the different models. The name of the 
subdirectory is used to identify the model(s) to use. Pass it as the argument `--name NAME` in the `evaluate.py` and 
`tag.py` scripts.

You can download our best model and best ensemble from the following [link][models_url].

The structure of the project should be:
```
.
├── my_model_1
│   └── model_weights.ckpt
├── my_model_2
│   └── model_weights.ckpt
├── my_ensemble_1
│   ├── model_weights_1.ckpt
│   ├── model_weights_2.ckpt
│   └── model_weights_3.ckpt
├── my_ensemble_2
│   ├── model_weights_1.ckpt
│   ├── model_weights_2.ckpt
│   ├── model_weights_3.ckpt
│   ├── model_weights_4.ckpt
│   └── model_weights_5.ckpt
├── ...
|
└── ...
```

[models_url]: https://www.dropbox.com/sh/92xtl13paxmwk1a/AACqeiYbhq7xCgHTOYjVYITca?dl=0
