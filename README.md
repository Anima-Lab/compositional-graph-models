# compositional-graph-models
Experiments in compositional learning with tree- and graph-structured models.

## Usage

To run training, run `poetry run train` in the root directory.
For configuration options, see `poetry run train --help` and
`compositional_graph_models/conf/training.yaml`.

To run evaluation use `poetry run evaluate data_path=... checkpoint_path=...
model_class=...` in the root directory, or change
`compositional_graph_models/conf/evaluation.yaml`.

## Notes to contributors
Please note that data JSON files are tracked through Git LFS, so take care not to commit any JSON logs your processes produce.
