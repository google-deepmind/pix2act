# Pix2Act

This repository contains code for [From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces](https://arxiv.org/abs/2306.00245).

## Setup and Prerequisites

You can clone the repository using `git clone https://github.com/google-deepmind/pix2act.git`.

### Environment

We recommend using `conda` or `venv` to manage dependencies for this project,
such as by following these commands:

```
conda create -n pix2act python=3.10
conda activate pixact
```

We recommend using Python 3.10 or later.

### Python Dependencies

First, install the platform-specific version of JAX based on instructions [here](https://jax.readthedocs.io/en/latest/installation.html).

You can then install the dependencies listed in `setup.py`. First,
`cd` to the directory of this repository, then run:

```
pip install -e ."[dev]"
```

### MiniWob Repository

Inference and evaluation for MiniWob also requires `miniwob-plusplus` (version 0.1), which is not covered by `setup.py`. We used a version of this library with local modifications for compatibility with Python 3. You can reproduce this setup using the following commands within a directory where you want to download this repository:

```
git clone https://github.com/stanfordnlp/miniwob-plusplus
cd miniwob-plusplus
git checkout bf8acbaa3c29b9553fef3cf107e9f236ef475f05
python3 -m pip install 2to3
2to3 python/miniwob -w
export PYTHONPATH=${PYTHONPATH}:${PWD}/python
```

### Protobuffers and gRPC

The code also uses protobuffers for storing MiniWob episodes (`tasks/miniwob/episode.proto`) and for communication
with a model server at inference time (`server/model_server.proto`). The `server/model_server.proto` file defines a [grpc service](https://grpc.io/docs/languages/python/generated-code/). Both protobuffers require code generation to generate Python modules corresponding to the `.proto` definitions:

```
pip install grpcio-tools
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=.   pix2act/server/model_server.proto
python -m grpc_tools.protoc -I . --python_out=.    pix2act/tasks/miniwob/episode.proto
```

### Chrome and ChromeDriver

Inference and evaluation also requires installing ChromeDriver. You can find
your current version of Chrome by running `google-chrome --version`, and then [download](https://chromedriver.chromium.org/downloads) the corresponding
ChromeDriver. The `chromedriver`
binary should be moved to `/usr/bin` or its location explicitly added your `PATH`.

NOTE: Differences in rendering (e.g. due to Chrome version or font
 configurations) between training and testing environments can have a
 significant effect on the performance.

We used Chrome version 110.0.05481.77 for our experiments. See Figure 5
in the appendix of the paper for an example rendering from the environment
used to tune the released checkpoints.

### Gin search path

The `PIX2STRUCT_INSTALL` variable contains the install location for Pix2Struct
which we will use to specify `--gin_search_paths`. If installed via `pip`, you can obtain its
value as:
```
PIX2STRUCT_INSTALL=$(pip show pix2struct | grep "Location" | cut -d ' ' -f 2)/pix2struct
```

### Google Cloud Authentication

Since the model checkpoints are hosted on Google Cloud, you might need to authenticate
before using this code.
```
gcloud auth application-default login
```

## Model Checkpoints

Here are finetuned model checkpoints for MiniWoB and WebShop:

Task    | GCS Path
------- | --------------------------------------------------
MiniWoB | `gs://pix2act-data/miniwob_mcts/checkpoint_332000`
WebShop | `gs://pix2act-data/webshop_bc/checkpoint_303600`

## Inference and Evaluation

### Model Server

You can start a model server for generating predictions for MiniWob via the following
command:

```
python -m pix2act.server.model_server_main \
  --alsologtostderr \
  --gin.CHECKPOINT_PATH="'${CKPT_PATH}'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 16}" \
  --gin.BATCH_SIZE=1 \
  --num_decodes=8 \
  --port=10000
```

Where `CKPT_PATH` points to a Pix2Struct model checkpoint, such as those
provided above. This will start a server that can be reached at
`localhost:10000`. We will assume that the environment variable `SERVER` points
to this address in the following documentation.

For WebShop, you can use the same command but with `--gin.TASK_FEATURE_LENGTHS="{'inputs': 4096, 'targets': 128}"` and `--num_decodes=10`.

The model server also supports inference on TPUs. There are various ways of
running the model server on TPUs. One option is to follow the instructions
here: https://github.com/google-research/pix2struct#setting-up-the-tpu

### Simple Inference

You can test the model server using `server/client_utils.py` by running:

```
python -m pix2act.server.simple_client \
  --alsologtostderr \
  --screenshot="${SCREENSHOT}" \
  --server="${SERVER}"
```

Where `SCREENSHOT` is a filepath to a png file that is provided to the model as input.

### Scaling

To run inference or evaluations more efficiently, it is recommended to start
multiple model servers and use client-side load balancing to balance requests
to this pool of model servers from multiple jobs running inference or evaluation
in parallel. Load balancing can be added to `client_utils.get_stub` based on
your configuration by following the instructions here:
https://grpc.io/blog/grpc-load-balancing/

This is particularly useful for running evaluations on all MiniWob tasks in
parallel or for generating episodes for training in parallel using
Monte-Carlo Tree Search.

### MiniWob Inference

First, start a local server to serve the MiniWob HTML files:

```
python3 -m http.server 8000 --directory ${MINIWOB_PATH}/html
```

Where `MINIWOB_PATH` points to a local directory containing the [`miniwob-plusplus`](https://github.com/Farama-Foundation/miniwob-plusplus/tree/v0.1) repository (see instructions above). We will assume that the variable `MINIWOB_URL` points to this server, e.g. `localhost:8000/html/miniwob`, in the following documentation.

You can then run inference for a given MiniWob seed and task using the following
script:

```
python -m pix2act.tasks.miniwob.analysis.run_episode \
  --alsologtostderr \
  --output=${OUTPUT} \
  --subdomain=${TASK} \
  --seed=${SEED} \
  --server=${SERVER} \
  --miniwob_url=${MINIWOB_URL}
```

Where `TASK` is a miniwob task (e.g. `enter-date`), `SEED` is a random seed for
 initializing the task environment (e.g. `1`), and `OUTPUT` is a path to write
 an HTML file for viewing the episode.

Similarly, we can run an evaluation for a given MiniWob task, computing the
success rate and mean reward, by running:

```
python -m pix2act.tasks.miniwob.eval.run_eval \
  --alsologtostderr \
  --output_dir=${OUTPUT_DIR} \
  --subdomain=${TASK} \
  --server=${SERVER} \
  --miniwob_url=${MINIWOB_URL} \
  --num_seeds=100
```

Here `OUTPUT_DIR` should be a directory for writing a json file with the
various statistics as well as HTML files for visualizing individual episodes,
assuming `--output_html` is true.

The `run_eval` script can be run in parallel for each MiniWob task (see the
scaling note above), with every task writing to the same `OUTPUT_DIR`. Then,
you can run:

```
python -m pix2act.tasks.miniwob.eval.aggregate_metrics \
  --alsologtostderr \
  --output_dir=${OUTPUT_DIR}
```

To aggregate and print statistics for all tasks.

### WebShop Inference

First, you must start a WebShop server. For instructions to start the server,
please see the official [WebShop repository](https://github.com/princeton-nlp/WebShop).
We will assume that `WEBSHOP_SERVER` points to the address of this server, e.g.
`localhost:3000`. Make sure you're able to access this via your browser before you proceed.

You can run evaluations on WebShop with the following command:

```
python -m pix2act.tasks.webshop.run_eval -- \
  --alsologtostderr \
  --webshop_url=${WEBSHOP_SERVER} \
  --output_dir=${OUTPUT_DIR} \
  --server=${SERVER}
```

Where `OUTPUT_DIR` specifies a location
for writing a json file with statistics and HTML files for visualizing
individual episodes.

## Training Data Generation

Training data for models is formatted as TensorFlow examples, following the
same format as Pix2Struct. These files contain a `image` field specifying the
bytes of the input image, and a `parse` field specifying the desired text
output.

### WebShop

First, ensure a WebShop server is running (see instructions above).

You should set `WEBSHOP_DATA_DIR` to an appropriate directory for storing
the generated data files. The command below assumes that local copy of the [WebShop repository](https://github.com/princeton-nlp/WebShop) is stored at `WEBSHOP_REPO`.

```
python -m pix2act.tasks.webshop.write_tf_examples \
  --alsologtostderr \
  --demo_file=${WEBSHOP_REPO}/baseline_models/data/il_trajs_finalized_images.jsonl \
  --human_goals_file=${WEBSHOP_REPO}/baseline_models/data/human_goals.json \
  --webshop_url=${WEBSHOP_SERVER} \
  --processed_dir=${WEBSHOP_DATA_DIR}
```

### MiniWob

We can generate episodes for training MiniWob models by running Monte-Carlo
Tree Search. First, you will need ensure both a MiniWob model server is running
for the policy, as well as a second model server for the value function
approximator, which we refer to as the critic. This can use the same command
as above for the policy. A checkpoint for the critic used in our experiments
is available at `gs://pix2act-data/miniwob_critic/checkpoint_306000`. We
will assume that `SERVER_CRITIC` refers to the location of this model server.
For clarity, we will assume that the policy model server is specified here
by `SERVER_POLICY`.

Below is an example command to run MCTS:

```
python -m pix2act.tasks.miniwob.search.run_mcts -- \
--alsologtostderr \
--output_dir=${OUTPUT_DIR} \
--server_critic=${SERVER_CRITIC} \
--server_policy=${SERVER_POLICY} \
--miniwob_url=${MINIWOB_URL} \
--task=${TASK} \
--max_episodes=10 \
--raise_exceptions
```

Where `TASK` is a miniwob task (e.g. `enter-date`). This script will write episodes as `Episode` protobuffers in the `MCTS_DIR`. See the note on scaling
above to run MCTS for efficiently. Multiple instances can write episodes
to the same `MCTS_DIR` in parallel. The script `tasks/miniwob/search/get_mcts_stat.py`
can aggregate and write statistics over the `Episode` protobuffers written by
these scripts. The script `tasks/miniwob/analysis/visualize_episodes.py` can be
used to visualize the generated episodes.

Given `Episode` protobuffers, we can convert these to TensorFlow examples
for model training:

```
python -m pix2act.tasks.miniwob.write_policy_tf_examples \
  --alsologtostderr \
  --input="${MCTS_DIR}/*.recordio" \
  --output_dir=${MINIWOB_DATA_DIR}
```

We can also generate training examples for training a new value function
approximator:

```
python -m pix2act.tasks.miniwob.search.write_value_fn_tf_examples \
  --alsologtostderr \
  --input="${MCTS_DIR}/*.recordio" \
  --output_dir=${MINIWOB_CRITIC_DATA_DIR}
```

Note that these two scripts use Beam so can process large datasets in parallel.

The script `miniwob/merge_tf_examples.py` can be useful for combining training
data from multiple sources.

## Model Training

The main experiments are implemented as a light wrapper around the
[T5X](https://github.com/google-research/t5x) library. For brevity, we
illustrate an example workflow of finetuning the pretrained base Pix2Struct. To run training on TPUs, please see to the T5X documentation.

Tasks for training WebShop and MiniWob models are specified in `pix2act_tasks.py`.
This file uses environment variables to specify data locations. The commands below
will then export the environment variables specifying these paths. Alternatively, locations
for training data can be defined directly by editing `pix2act_tasks.py`. Follow the
instructions above for populating data at these paths.

### WebShop

```
export WEBSHOP_DATA_DIR
python -m t5x.train \
  --gin_search_paths="pix2act/configs,$PIX2STRUCT_INSTALL" \
  --gin_file="configs/models/pix2struct.gin" \
  --gin_file="runs/train.gin" \
  --gin_file="configs/sizes/base.gin" \
  --gin_file="configs/optimizers/adafactor.gin" \
  --gin_file="schedules/webshop.gin" \
  --gin_file="init/miniwob_100_buckets_base_init.gin" \
  --gin.MIXTURE_OR_TASK_NAME="'webshop'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 4096, 'targets': 128}" \
  --gin.BATCH_SIZE=32
```

Where `MODEL_DIR` points to a directory to write model checkpoints and other
files generated by model training. Note that the above command initializes the
model parameters from a MiniWoB checkpoint as described in Section 5.1 of our
paper. Alternatively, you can train a model from a Pix2Struct checkpoint without
any intermediate finetuning on MiniWob by setting `--gin_file="init/pix2struct_base_init.gin"`.

### MiniWoB

Similarly, we can train a MiniWob policy model:

```
export MINIWOB_DATA_DIR
python -m t5x.train \
  --gin_search_paths="pix2act/configs,$PIX2STRUCT_INSTALL" \
  --gin_file="configs/models/pix2struct.gin" \
  --gin_file="runs/train.gin" \
  --gin_file="configs/sizes/base.gin" \
  --gin_file="configs/optimizers/adafactor.gin" \
  --gin_file="schedules/miniwob.gin" \
  --gin_file="configs/init/pix2struct_base_init.gin" \
  --gin.MIXTURE_OR_TASK_NAME="'miniwob'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 16}" \
  --gin.BATCH_SIZE=512
```

Or a MiniWob critic model:

```
export MINIWOB_CRITIC_DATA_DIR
python -m t5x.train \
  --gin_search_paths="pix2act/configs,$PIX2STRUCT_INSTALL" \
  --gin_file="configs/models/pix2struct.gin" \
  --gin_file="runs/train.gin" \
  --gin_file="configs/sizes/base.gin" \
  --gin_file="configs/optimizers/adafactor.gin" \
  --gin_file="schedules/miniwob.gin" \
  --gin_file="configs/init/pix2struct_base_init.gin" \
  --gin.MIXTURE_OR_TASK_NAME="'miniwob_critic'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 16}" \
  --gin.BATCH_SIZE=512
```

## Cursor Graphics

The default argument for `--cursor_dir` for several of the scripts above is `gs://pix2act-data/cursors/`, which includes the cursor graphics used to
train our models. The cursor graphics are from
[yaru](https://github.com/ubuntu/yaru). Please see the corresponding [readme](https://github.com/ubuntu/yaru/tree/master/icons#copying-or-reusing) and [license](https://creativecommons.org/licenses/by-sa/4.0/). These are used to render cursors on top of the screenshots
generated via Selenium.

## Citation

If you are using this library, please cite:

```
@inproceedings{
  shaw2023pixels,
  title={From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces},
  author={Shaw, Peter and Joshi, Mandar and Cohan, James and Berant, Jonathan and Pasupat, Panupong and Hu, Hexiang and Khandelwal, Urvashi and Lee, Kenton and Toutanova, Kristina},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
  url={https://arxiv.org/abs/2306.00245}
}
```

## Note

*This is not an officially supported Google product.*
