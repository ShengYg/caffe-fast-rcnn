#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {
//normalize along channel
template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data1, const Dtype* data2, Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      Dtype tdata1 = data1[(n * channels + c) * spatial_dim + s];
      Dtype tdata2 = data2[(n * channels + c) * spatial_dim + s];
      dot += tdata1 * tdata2;
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
__global__ void kernel_channel_norm(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      Dtype tdata = data[(n * channels + c) * spatial_dim + s];
      dot += tdata * tdata;
    }
    channel_dot[index] = sqrt(dot) + 1e-6;
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] = channel_max[n * spatial_dim + s] - data[index];
  }
}

template <typename Dtype>
__global__ void kernel_channel_mul(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] *= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // Dtype* scale_data = scale_.mutable_gpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  int count = bottom[0]->count();
  // int dim = count / num;
  int dimScale = height * width;

  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  kernel_channel_norm<Dtype><<<CAFFE_GET_BLOCKS(num * dimScale),
    CAFFE_CUDA_NUM_THREADS>>>(num, channels, dimScale, 
      bottom[0]->gpu_data(), norm_.mutable_gpu_data());

  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, dimScale, 
      norm_.gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  int count = bottom[0]->count();
  // int dim = count / num;
  int dimScale = height * width;

  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(num * dimScale),
    CAFFE_CUDA_NUM_THREADS>>>(num, channels, dimScale, 
      top[0]->gpu_data(), top[0]->gpu_diff(), sum_channel_.mutable_gpu_data());

  kernel_channel_mul<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, dimScale, 
      sum_channel_.gpu_data(), top[0]->mutable_gpu_data());

  caffe_gpu_sub(count, top[0]->gpu_diff(), 
    top[0]->gpu_data(), bottom[0]->mutable_gpu_diff());

  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, dimScale, 
      norm_.gpu_data(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);


}  // namespace caffe