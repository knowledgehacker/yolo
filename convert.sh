#!/bin/sh

# before convert, pleas download .cfg and .weights from https://pjreddie.com/darknet/imagenet/,
# and put .cfg file in "cfg" directory and .weights in "data/weights" directory

DIR=data/weights

#NET=darknet19
NET=darknet19_448
# by default the model instead of weights are saved, to enbale load_weights, pass argument "--weights_only"
python3 convert.py --weights_only --plot_model cfg/${NET}_-.cfg ${DIR}/${NET}.weights ${DIR}/${NET}.h5

#NET=extraction
# by default the model instead of weights are saved, to enbale load_weights, pass argument "--weights_only"
#python3 convert.py --weights_only --plot_model cfg/${NET}_-.cfg ${DIR}/${NET}.weights ${DIR}/${NET}.h5

