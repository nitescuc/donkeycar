export MODEL=~/Downloads/mypilot.h5
#export MODEL=./models/sim-linear-vanilla-5star.h5
python3 manage.py drive --model=$MODEL --type=linear
