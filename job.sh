echo "Cleaning all past files"
make cleanall
make
if [ $? -eq 0 ];
  then
  qsub launch.pbs
  if [ $? -eq 0 ];
  	then
  	echo "Launch to server"
  fi
fi
