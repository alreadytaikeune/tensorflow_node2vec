TF_CFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
FLAGS:=-DNO_SHARDER=1
# FLAGS:=

CC=g++-5

SRCS=$(wildcard *.cc)
OBJS=$(patsubst %.cc,%.o,$(SRCS))
TARG=$(patsubst %.o,%,$(SRCS))

all: test_graph_reader test_graph_types

%.o: %.cc
	$(CC) -fPIC $(TF_CFLAGS) $(FLAGS) -O2 -std=c++11 -I/usr/local/include -I.. -c $< -o $@

test_graph_reader: test_graph_reader.o
	$(CC) -Wl,--no-as-needed $(TF_LFLAGS) -L../.. -o $@ $^ -lgraphseq_ops  -lboost_system -lboost_filesystem 

test_graph_types: test_graph_types.o
	$(CC) -Wl,--no-as-needed $(TF_LFLAGS) -L../.. -o $@ $^ -lgraphseq_ops  -lboost_system -lboost_filesystem 