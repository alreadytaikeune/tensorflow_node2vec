
TF_CFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

all: node2vec_ops.so randwalk_ops.so


test: test.o graphml.o
	g++ test.o graphml.o -o test

%.o: %.cc
	g++ -fPIC $(TF_CFLAGS) -O2 -std=c++11 -I/usr/local/include -c $< -o $@

node2vec_ops.so: graphml.o node2vec_kernel.o node2vec_ops.o
	g++ -shared -Wl,--no-as-needed $(TF_LFLAGS) -o $@ $^

randwalk_ops.so: graphml.o random_walk_kernel.o random_walk_ops.o
	g++ -shared -Wl,--no-as-needed $(TF_LFLAGS) -o $@ $^