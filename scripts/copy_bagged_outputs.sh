#!/usr/bin/env bash
# cd /data/eth/fs18_machine-perception/mp18-eye-gaze-estimation/outputs/DenseBag/
# rsync -avPr ipercept@ipercept:/home/ipercept/iPercept_marcel/outputs/DenseBag_RS* .
# rsync -avPr ipercept@ipercept2:/home/ipercept/iPercept_marcel/outputs/DenseBag_RS* .
# rsync -avPr ipercept@ipercept3:/home/ipercept/iPercept_marcel/outputs/DenseBag_RS* .
#

 cd /data/eth/fs18_machine-perception/mp18-eye-gaze-estimation/outputs/DenseBagValidation/
 rsync -avPr ipercept@ipercept:/home/ipercept/iPercept_marcel/outputs/DenseBag_Validation_RS* .
 rsync -avPr ipercept@ipercept2:/home/ipercept/iPercept_marcel/outputs/DenseBag_Validation_RS* .
 rsync -avPr ipercept@ipercept3:/home/ipercept/iPercept_marcel/outputs/DenseBag_Validation_RS* .
