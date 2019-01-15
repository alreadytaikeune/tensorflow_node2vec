#include "graph_kernel_base.h"

BaseGraphKernel::BaseGraphKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx){
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    num_threads_ = worker_threads.num_threads;
    guarded_philox_.Init(0, 0);
  }


void BaseGraphKernel::Compute(OpKernelContext* ctx) {
    Tensor epoch(DT_INT32, TensorShape({}));
    Tensor total(DT_INT32, TensorShape({}));
    Tensor nb_valid_nodes(DT_INT32, TensorShape({}));
    Tensor walk(DT_INT32, TensorShape({seq_size_}));
    {
      mutex_lock l(mu_);
      NextWalk(ctx, walk);
      epoch.scalar<int32>()() = current_epoch_;
      total.scalar<int32>()() = total_seq_generated_;
    }
    nb_valid_nodes.scalar<int32>()() = nb_valid_nodes_;
    ctx->set_output(0, node_id_);
    ctx->set_output(1, walk);
    ctx->set_output(2, epoch);
    ctx->set_output(3, total);
    ctx->set_output(4, nb_valid_nodes);

  }


void BaseGraphKernel::NextWalk(OpKernelContext* ctx, Tensor& walk) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int N = valid_nodes_.size();
    int available = (write_walk_idx + PRECOMPUTE - cur_walk_idx) % PRECOMPUTE;
    if(available <= LOW_WATER_MARK){
      int start = write_walk_idx;
      int end = cur_walk_idx-1;
      if(end <= start)
        end += PRECOMPUTE;
      auto fn = [this](int64 s, int64 e){
        PrecomputeWalks(write_walk_idx, s, e);
      };

      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      // worker_threads.num_threads
      #ifndef NO_SHARDER
      Shard(1, worker_threads.workers,
            end-start, 20000000,
            fn);
      #else
        PrecomputeWalks(write_walk_idx, start, end);
      #endif
      write_walk_idx+=(end-start);
      write_walk_idx%=PRECOMPUTE;
      current_node_idx_ += (end-start);
      current_node_idx_ %= N;

    }

    auto w = walk.flat<int32>();
    w = precomputed_walks.matrix<int32>().chip<0>(cur_walk_idx);
    cur_walk_idx++;
    total_seq_generated_++;
    cur_walk_idx%=PRECOMPUTE;
    current_epoch_ = total_seq_generated_/N;
  }


void BaseGraphKernel::PrecomputeWalks(int write_idx, int start_idx, int end_idx){
    int N = valid_nodes_.size();
    random::PhiloxRandom phi = guarded_philox_.ReserveSamples128(seq_size_*2);  // thread safe
    random::SimplePhilox gen(&phi);
    for(int i=start_idx; i<end_idx; i++){
      PrecomputeWalk((write_idx+i)%PRECOMPUTE, valid_nodes_[(current_node_idx_+i)%N], gen);
    }
}