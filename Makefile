TF_CFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
FLAGS:=-DNO_SHARDER=1
# FLAGS:=

SRC_DIR=cc
CC=g++-5

SRCS=$(wildcard cc/*.cc)
OBJS=$(patsubst %.cc,%.o,$(SRCS))

all: graphseq_ops.so

test: test.o graphml.o
	$(CC) test.o graphml.o -o test

%.o: %.cc
	$(CC) -fPIC $(TF_CFLAGS) $(FLAGS) -O2 -std=c++11 -I/usr/local/include -c $< -o $@

graphseq_ops.so: $(OBJS)
	$(CC) -shared -Wl,--no-as-needed $(TF_LFLAGS) -o $@ $^
