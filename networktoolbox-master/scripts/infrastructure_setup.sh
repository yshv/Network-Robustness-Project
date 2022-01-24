
nodes=("mammoth" "lyon" "vienna" "verona" "monaco" "budapest")
for i in "${nodes[@]}"
        do
                ssh $i tmux new -d -s ssh ssh -L 27019:localhost:27111 amsterdam

        done

