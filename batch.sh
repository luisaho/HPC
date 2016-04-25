#!/usr/bin/env zsh
 
### Job name
#BSUB -J lect0012serial
 
### File / path where STDOUT & STDERR will be written
###    %J is the job ID
#BSUB -o lect0012serial.%J

### Project lect0012 for higher priority on CPU nodes
#BSUB -P lect0012

### Job with X11-Forwarding (e.g. for using tools)
### #BSUB -XF

### Advanced reservation: Uncomment one of these for 
### interactive usage during Software Lab sessions
### Naming: lect0012_<month><day>
### #BSUB -U lect0012_0308
### #BSUB -U lect0012_0318
#BSUB -U lect0012_0421
### #BSUB -U lect0012_0428
### #BSUB -U lect0012_0512
### #BSUB -U lect0012_0602
### #BSUB -U lect0012_0609
 
### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 0:10
 
### Request memory you need for your job in TOTAL in MB
#BSUB -M 512

### Request hardware and parallelism that you need
### For serial CPU usage use mpi-s nodes 
### #BSUB -m mpi-s
### For parallel CPU usage use mpi-s nodes (or BCS with "-a bcs")
### and set parallelism
### #BSUC -m mpi-s
### #BSUB -a openmp
### #BSUB -n 12
### For GPU usage use -a or -q gpu (see https://doc.itc.rwth-aachen.de)
#BSUB -a gpu
#BSUB -R fermi
 
### Change to the work directory
cd $HOME

### Change compiler if needed/ load modules 
### module switch intel pgi

### Execute your application
make
make run
