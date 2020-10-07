run_dir=run

make

cp turbChannel.par $SCRATCH/smarties/$run_dir
cp turbChannel.ma2 $SCRATCH/smarties/$run_dir
cp turbChannel.re2 $SCRATCH/smarties/$run_dir

smarties.py $SMARTIES_ROOT/contrib/nek5000 settings.json --runname $run_dir 

