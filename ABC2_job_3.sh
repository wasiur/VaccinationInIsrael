source activate dynamic_survival_analysis

DATAFILE="aggregate_data_Mar11.csv"
OUTPUTFOLDER="DSA_ABC_server_segment3_50"
DAYZERO="2021-02-15"
FINALDATE="2021-02-28"
SEGMENT=3
ACCEPT=0.1
N=100000
T=60
THREADS=230

if [ -d "$OUTPUTFOLDER" -a ! -h "$OUTPUTFOLDER" ]
then
  echo "Output folder exits."
else
    echo "Output folder does not exist...creating one"
    mkdir $OUTPUTFOLDER
fi


mpiexec -n $THREADS python ABC_2dose_MPI.py -d $DATAFILE -o $OUTPUTFOLDER --day-zero=$DAYZERO --final-date=$FINALDATE -N $N -T $T --plot --segment=$SEGMENT --accept=$ACCEPT

