# set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 600, 400 
# set output 'binary.1.png'
NS=ARG1
NA=ARG2
FILE=ARG3
ICOL=ARG4+0
NL=(NA*NA+NA)/2
NCOL=3+NS+NA+2*NA
print NS
print NA
print FILE
print ICOL
print NL
print NCOL
plot FILE binary array=(NCOL) format='%float32' using ICOL with lines
