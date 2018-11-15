#! /bin/bash
for f in *.dat
  do
    gnuplot -e "filename='${f%.*}'" plot
  done
for f in *.eps
  do
  convert -type Grayscale -density 600 -flatten ${f%.*}.eps ${f%.*}.png
  rm ${f%.*}.eps
done
