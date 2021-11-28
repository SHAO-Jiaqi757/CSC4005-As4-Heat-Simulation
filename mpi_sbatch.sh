#!/bin/bash
chmod +x ./build/*
tag="mpi"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
export LD_LIBRARY_PATH=./build/
rm -r tmp
mkdir -p tmp
for i in {1..128..8}
do
	for size in {600..1000..100}
	do
	    line=$(squeue --me | wc -l)
	    while [ $line -gt 10 ]
	    do
		line=$(squeue --me | wc -l)
		echo "$line jobs in squeue"
		sleep 2s
	    done
	    echo "using $i cores"
	    echo "#!/bin/bash" > ./tmp/$size.sh
	    echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/$size.sh
	    echo "mpirun -n $i ./build/mpi $size 4 >> ./result/logs/${tag}-${dt}.log" >> ./tmp/$size.sh
	    cat ./tmp/$size.sh
	    sbatch --wait --account=csc4005 --partition=debug --qos=normal --nodes=4 --ntasks=$i ./tmp/$size.sh
	done
done
