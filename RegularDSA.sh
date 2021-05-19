source activate dynamic_survival_analysis


DATAFILE="DSA_data.csv"
OUTPUTFOLDER="RegularDSAPlots_Sep1Nov1"
LOCATION="Israel"
DAYZERO="2020-09-01"
FINALDATE="2020-11-01"
NITER=15000
NCHAINS=1

T=240

mkdir $OUTPUTFOLDER

time python DSA_Bayesian.py -d $DATAFILE -o $OUTPUTFOLDER --day-zero=$DAYZERO --final-date=$FINALDATE --niter=$NITER --nchains=$NCHAINS -T $T --location=$LOCATION
