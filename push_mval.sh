#!/bin/bash
SRC=$1
DST=$2

if [[ $? -eq 3 ]]; then
    rsync --exclude='.git/' --exclude='model_*/' --progress -r \
          -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
          "$SRC" \
          "$DST" ;
    exit
fi

echo Pulling form "$SRC" to "$DST"

for box in {2..4}; do
    echo Box $box
    ssh mval${box} "mkdir -p $(dirname $DST) 1>/dev/null"
    rsync --exclude='.git/' --exclude='model_*/' --progress -r \
        -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        "$SRC" \
        mval${box}:${DST} ;
done

rsync --exclude='.git/' --progress -r -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    ~/repos/progressive-gan/ mval1:repos/progressive-gan
