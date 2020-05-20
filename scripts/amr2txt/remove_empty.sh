#!/bin/bash

for dir in *
do
  if [[ -d "$dir" ]];
  then
    empty=1
    for i in {1..15};
    do
      if [[ -f "$dir/checkpoint_mymodel_$i.pth" ]];
      then
        empty=0
      fi
    done
    if [ "$empty" == 1 ];
    then
      echo "Removing $dir"
      rm -rf $dir
    fi
  fi

done
