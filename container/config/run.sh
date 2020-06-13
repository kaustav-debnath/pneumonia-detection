#!/bin/bash
# #!/bin/sh

echo Inside run.sh
echo $1
if [[ "$1" = train ]]
then
    echo $PWD
    chmod +x train.py
    echo inside train
    python train.py
  else
    echo $PWD
    echo inside serve
    chmod +x predictor.py
    echo $(pwd)
    python predictor.py 
fi
