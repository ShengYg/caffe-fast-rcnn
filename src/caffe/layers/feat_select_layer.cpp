#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <stdlib.h>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/feat_select_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void FeatSelectLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[0]->height(), 1);
		
		CHECK_EQ(bottom[1]->channels(), 1);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		CHECK_EQ(bottom[1]->num(), bottom[0]->num());
	}

	template <typename Dtype>
	void FeatSelectLayer<Dtype>::Reshape(
    	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		int num = bottom[0]->num();
		int out_num = (num * num + num) / 2;

		top[0]->Reshape(out_num, bottom[0]->channels(), 1, 1);
		top[1]->Reshape(out_num, 1, 1, 1);
	}

	template <typename Dtype>
	void FeatSelectLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_label = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* top_label = top[1]->mutable_cpu_data();
		int dim = bottom[0]->channels();

		caffe_set(top[1]->count(), Dtype(0), top_label);
		int k = 0;
		for(int i = 0; i < bottom[0]->num(); i++) {
			for(int j = i; j < bottom[0]->num(); j++){
				caffe_sub(dim, bottom_data + i * dim, bottom_data + j * dim, top_data + k * dim);
				if (static_cast<int>(bottom_label[i]) == static_cast<int>(bottom_label[j]))
					top_label[k] = Dtype(1);
				k++;
			}
		}
	}

	template <typename Dtype>
	void FeatSelectLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		int dim = bottom[0]->channels();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		
		if(propagate_down[0]){
			int k = 0;
			for(int i = 0; i < bottom[0]->num(); i++) {
				for(int j = i; j < bottom[0]->num(); j++){
					caffe_cpu_axpby(
							dim,
							Dtype(1.0),
							top[0]->cpu_diff() + (k * dim),
							Dtype(1.0),
							bottom_diff + i*dim
							);
					caffe_cpu_axpby(
							dim,
							-Dtype(1.0),
							top[0]->cpu_diff() + (k * dim),
							Dtype(1.0),
							bottom_diff + j*dim
							);
					k++;
				}
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(FeatSelectLayer);
#endif

INSTANTIATE_CLASS(FeatSelectLayer);
REGISTER_LAYER_CLASS(FeatSelect);
}//namespace caffe