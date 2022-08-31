#!/bin/bash

# make coldens
# qsub launch_coldens.pbs
# cat coldens.out

step=1
init=0
end=219
# for value in {0..100..10}
for value in $(seq $init $step $end)
do
	# echo "/outmin = $value/outmin = $value"
	sed -i "s/outmin = .*/outmin = $value/g" source/coldens.f90
	# echo "/outmax = $value/outmax = $(($value + $step-1))"
	sed -i "s/outmax = .*/outmax = $(($value + $step-1))/g" source/coldens.f90

	make coldens
    mv coldens coldens_$value
	qsub -F $value launch_coldens.pbs
	
done
#sed -i 's/outmin = 0/outmin = 100/g' get_arguments.f90


# Para borrar todos los jobs de golpe
# qselect -u jsmendezh | xargs qdel
