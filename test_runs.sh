#!/bin/bash

let num_matches="$1"
p_script="$2"
let num_errors=0
let num_wins=0
let num_losses=0
let num_ties=0

for ((i=1; i<=$num_matches; i++))
do
	if (( $(( i%2 )) == 0 )) 
	then
		python "$p_script" -l RANDOM -b as-pacman-ghost-riders/myTeam.py > "./as-pacman-ghost-riders/play_logs/$i.log"
		error="$?"
		echo "Playing blue"
		echo "Error code: $error" 
		if [ $error -ne 0 ] 
		then
			if [ $error -eq 10 ]
			then
				echo "updating wins"
				((num_wins++))
				echo "BLUE WINS!!!"
			elif [ $error -eq 11 ]
			then 
				echo "updating losses"
				((num_losses++))
			else
				echo "updating errors"
				echo "Crashed on play $i"
				((num_errors++))
				exit -1
			fi
		else
			echo "updating ties"
			((num_ties++))
		fi
	else
		python "$p_script" -l RANDOM -r as-pacman-ghost-riders/myTeam.py > "./as-pacman-ghost-riders/play_logs/$i.log"
		error="$?"
		echo "Playing red"
		echo "Error code: $error"
		if [ $error -ne 0 ] 
		then
			if [ $error -eq 11 ]
			then
				echo "updating wins"
				((num_wins++))
			elif [ $error -eq 10 ]
			then 
				echo "updating losses"
				((num_losses++))
			else
				echo "updating errors"
				echo "Crashed on play $i"
				((num_errors++))
				exit -1
			fi
		else
			echo "updating ties"
			((num_ties++))
		fi
	fi
	
done

echo "Total number of errors is: $num_errors"
echo "Total number of wins is: $num_wins"
echo "Total number of losses is: $num_losses"
echo "Total number of ties is: $num_ties"
