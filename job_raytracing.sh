#!/bin/bash
step=15
init=0
end=219
# for value in {0..100..10}
for value in $(seq $init $step $end)
do
	# echo "/outmin = $value/outmin = $value"
	sed -i "s/outmin = .*/outmin = $value/g" source/raytracing.f90
	# echo "/outmax = $value/outmax = $(($value + $step-1))"
	sed -i "s/outmax = .*/outmax = $(($value + $step-1))/g" source/raytracing.f90

	make raytracing_c
    mv raytracing_c raytracing_$value
	qsub -F $value launch_xray.pbs
	
done
#sed -i 's/outmin = 0/outmin = 100/g' get_arguments.f90


# Para borrar todos los jobs de golpe
# qselect -u jsmendezh | xargs qdel
