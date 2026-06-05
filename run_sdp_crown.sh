# Reproducing Talbe 1 in the SDP-CROWN paper
# MNIST Models
./venv_sdp/bin/python sdp_crown.py --model mnist_mlp --radius 1.0
./venv_sdp/bin/python sdp_crown.py --model mnist_convsmall --radius 0.3
./venv_sdp/bin/python sdp_crown.py --model mnist_convlarge --radius 0.3

# CIRAR-10 Models
./venv_sdp/bin/python sdp_crown.py --model cifar10_cnn_a --radius 24/255
./venv_sdp/bin/python sdp_crown.py --model cifar10_cnn_b --radius 24/255
./venv_sdp/bin/python sdp_crown.py --model cifar10_cnn_c --radius 24/255
./venv_sdp/bin/python sdp_crown.py --model cifar10_convsmall --radius 24/255
./venv_sdp/bin/python sdp_crown.py --model cifar10_convdeep --radius 24/255
./venv_sdp/bin/python sdp_crown.py --model cifar10_convlarge --radius 8/255