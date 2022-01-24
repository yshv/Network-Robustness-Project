length=200
step=50
processes=$((length/step))
counter=0
while [ $counter -lt $processes ]
do
#  echo $counter
  start=$((counter*step))
#  echo $start
  stop=$((start+step))
#  echo $stop
  python NetworkToolkit/NetworkSimulator.py -start $start -stop $stop &
#  tmux next-window
  ((counter++))
done
