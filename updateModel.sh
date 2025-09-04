#!/bin/bash

cd ./code/decoder

# Ask user for subject number (e.g., 0)
read -p "subjectID number: " subID

# Prepend 'e' to make full subject ID
subjectID="e$subID"

# Run MATLAB with subject ID
matlab -nodisplay -nosplash -r "computeModel('$subjectID'); exit"
