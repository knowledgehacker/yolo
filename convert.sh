#!/bin/sh

# before convert, pleas download .cfg and .weights from https://pjreddie.com/darknet/imagenet/,
# and put .cfg file in "cfg" directory and .weights in "data/weights" directory

NET=darknet19

# by default the model instead of weights are saved, to enbale load_weights, pass argument "--weights_only"
python3 convert.py --weights_only cfg/${NET}_-2.cfg data/weights/${NET}.weights data/weights/${NET}.h5
