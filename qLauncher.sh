#!/bin/bash
read -p "subjectID: " subID
read -p "[d] Demographics, [c] Color Blindness Test, or [t] TLI? " mode

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )" #${BASH_SOURCE[0]} finds path to currently executing script and >/dev/null discards any output from cd
echo $dir

#########################################################################

if [ "$mode" = "d" ]
	then
 	echo "
----------------------------------
----> Demographics <-----
----------------------------------
"

foldername=$dir/data/"e"$subID"_"$(date +%Y%m%d) 
mkdir -p $foldername

python3 ./visualInterface/demographics.py "subject"$subID "$foldername"


#########################################################################
elif [ "$mode" = "c" ]
 	then
 	echo "
----------------------------------
----> Color Blindness Test <-----
----------------------------------
"
foldername=$dir/data/"e"$subID"_"$(date +%Y%m%d) 
mkdir -p $foldername

python3 ./visualInterface/color_gap_test_gui.py "subject"$subID "$foldername"

#########################################################################
elif [ "$mode" = "t" ]
 	then
 	echo "
----------------------------------
----> NASA Task Load Index <-----
----------------------------------
"
foldername=$dir/data/"e"$subID"_"$(date +%Y%m%d) 
mkdir -p $foldername

python3 ./visualInterface/NASA_TLI.py "subject"$subID "$foldername"


fi #ends the if statement 