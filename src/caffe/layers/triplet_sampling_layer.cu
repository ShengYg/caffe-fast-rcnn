#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triplet_sampling_layer.hpp"

namespace caffe {
template <typename Dtype>
void TripletSamplingLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Forward_cpu(bottom, top);
}

template <typename Dtype>
void TripletSamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	Backward_cpu(top, propagate_down, bottom);
}
INSTANTIATE_LAYER_GPU_FUNCS(TripletSamplingLayer);
}//namespace caffe