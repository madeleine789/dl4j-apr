#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J APRtestjob
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=8
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=10GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=24:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A ap
## Specyfikacja partycji
#SBATCH -p plgrid
## Plik ze standardowym wyjściem
#SBATCH --output="output-%j.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="error-%j.err"

cd $SLURM_SUBMIT_DIR

module load plgrid/tools/java8/1.8.0_60
module load plgrid/tools/maven/3.3.9
module load plgrid/libs/mkl/2017.0.2

export JAVA_HOME=/net/software/local/software/Java/1.8.0_60/
export PATH=$PATH:$JAVA_HOME/bin:/net/software/local/maven/3.3.9/bin
export OMP_NUM_THREADS=8

mkdir -p $SCRATCH/repo/ && export LOCAL_REPO=$SCRATCH/repo/

cd $SCRATCH/dl4j-apr/ && mvn -U -DskipTests clean install exec:java -Dmaven.repo.local=$LOCAL_REPO
