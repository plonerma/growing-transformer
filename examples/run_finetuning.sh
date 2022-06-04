#!/usr/bin/env bash

nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=cola lr=8e-5,5e-5,3e-5,1e-5 epochs=3,5,10 batch=16,32 load=$1 device=$2
nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=sst2 lr=8e-5,5e-5,3e-5,1e-5 epochs=3,5,10 batch=16,32 load=$1 device=$2
nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=mrpc lr=8e-5,5e-5,3e-5,1e-5 epochs=3,5,10 batch=16,32 load=$1 device=$2
nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=stsb lr=8e-5,5e-5,3e-5,1e-5 epochs=3,5,10 batch=16,32 load=$1 device=$2
nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=rte lr=8e-5,5e-5,3e-5,1e-5 epochs=3,5,10 batch=16,32 load=$1 device=$2
nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=wnli lr=8e-5,5e-5,3e-5,1e-5 epochs=3,5,10 batch=16,32 load=$1 device=$2

nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=qqp lr=8e-5,5e-5 epochs=5 batch=32 load=$1 device=$2
nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=mnli lr=8e-5,5e-5 epochs=5 batch=32 load=$1 device=$2
nice -15 python examples/finetune.py --multirun save=false checkpoint_every=0 task=qnli lr=8e-5,5e-5 epochs=5 batch=32 load=$1 device=$2
