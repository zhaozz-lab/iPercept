#!/usr/bin/env bash
 cd /data/eth/fs18_machine-perception/mp18-eye-gaze-estimation/outputs
 rsync -avPr ipercept@ipercept:/home/ipercept/iPercept_marcel/outputs/DenseNetBagged_RS* .
 rsync -avPr ipercept@ipercept2:/home/ipercept/iPercept_marcel/outputs/DenseNetBagged_RS* .
 rsync -avPr ipercept@ipercept3:/home/ipercept/iPercept_marcel/outputs/DenseNetBagged_RS* .
