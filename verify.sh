# loop over the filenames of libSVm datasets
for f in a1a a2a a3a a4a diabetes ionosphere_scale
do
  # loop over different epsilons
  for e in 5 3 1 0.5
  do
    echo "Processing dataset $f with eps=$e"
    time ./main -a 1 -e $e $f
  done
done
