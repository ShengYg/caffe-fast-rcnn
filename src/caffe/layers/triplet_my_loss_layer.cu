#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triplet_my_loss_layer.hpp"

using namespace std;
using namespace cv;

namespace caffe {

template <typename Dtype>
void TripletMyLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	Forward_cpu(bottom, top);
}

template <typename Dtype>
void TripletMyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	Backward_cpu(top, propagate_down, bottom);
}

template <typename Dtype>
void TripletMyLossLayer<Dtype>::set_mask_gpu(const vector<Blob<Dtype>*>& bottom){
}

// template <typename Dtype>
// void TripletMyLossLayer<Dtype>::set_mask_gpu(const vector<Blob<Dtype>*>& bottom){
// 	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();
// 	int hard_num = triplet_my_loss_param.hard_num();
// 	int rand_num = triplet_my_loss_param.rand_num();
// 	float margin = triplet_my_loss_param.margin();
// 	int bg_label = triplet_my_loss_param.bg_label();
// 	int neg_num = hard_num + rand_num;

// 	const Dtype* bottom_data = bottom[0]->gpu_data();
// 	const Dtype* label = bottom[1]->cpu_data();
// 	int num = bottom[0]->num();
// 	int dim = bottom[0]->count() / bottom[0]->num();
// 	Dtype* dis_data = dis_.mutable_cpu_data();
// 	Dtype* dis_data_gpu = dis_.mutable_gpu_data();
// 	Dtype* mask_data = mask_.mutable_cpu_data();

// 	caffe_gpu_set(dis_.count(), Dtype(0.0), dis_data_gpu);
// 	caffe_set(mask_.count(), Dtype(0.0), mask_data);
// 	for(int i = 0; i < num; i ++){
// 		for(int j = i + 1; j < num; j ++){
// 			caffe_gpu_sub(dim,
// 					bottom_data + i*dim,
// 					bottom_data + j*dim,
// 					diff_ap_.mutable_gpu_data());
// 			caffe_gpu_dot(dim,
// 					diff_ap_.gpu_data(),
// 					diff_ap_.gpu_data(),
// 					dis_data_gpu + i * num + j);
// 			caffe_gpu_memcpy(1, dis_data_gpu + i * num + j, dis_data_gpu + j * num + i);
// 		}
// 	}
// 	//select samples
// 	vector<pair<float, int> >negpairs;
// 	vector<int> sid1;
// 	vector<int> sid2;
// 	for(int i = 0; i < num; i++){
// 		if(static_cast<int>(label[i]) == bg_label)
// 			continue;
// 		for(int j = i + 1; j < num; j++){
// 			if(label[i] == label[j]){
// 				negpairs.clear();
// 				sid1.clear();
// 				sid2.clear();
// 				for(int k = 0; k < num; k++){
// 					if(static_cast<int>(label[k]) == bg_label)
// 						continue;
// 					if(label[k] != label[i]){
// 						Dtype tloss = max(Dtype(0.0), dis_data[i * num + j] - dis_data[i * num + k] + Dtype(margin));
// 						if(tloss != 0) 
// 							negpairs.push_back(make_pair(dis_data[i * num + k], k));
// 					}
// 				}
// 				if(negpairs.size() <= neg_num){
// 					for(int k = 0; k < negpairs.size(); k++){
// 						int id = negpairs[k].second;
// 						mask_data[i * num * num + j * num + id] = 1;
// 					}
// 				}
// 				else{
// 					sort(negpairs.begin(), negpairs.end());

// 					for(int k = 0; k < neg_num; k++){
// 						sid1.push_back(negpairs[k].second);
// 					}
// 					for(int k = neg_num; k < negpairs.size(); k++){
// 						sid2.push_back(negpairs[k].second);
// 					}
// 					std::random_shuffle(sid1.begin(), sid1.end(), myrandom);
// 					for(int k = 0; k < min(hard_num, (int)(sid1.size()) ); k++){
// 						mask_data[i * num * num + j * num + sid1[k]] = 1;
// 					}
// 					for(int k = hard_num; k < sid1.size(); k ++){
// 						sid2.push_back(sid1[k]);
// 					}
// 					std::random_shuffle(sid2.begin(), sid2.end(), myrandom);
// 					for(int k = 0; k < min(rand_num, (int)(sid2.size()) ); k++){
// 						mask_data[i * num * num + j * num + sid2[k]] = 1;
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// template <typename Dtype>
// void TripletMyLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {

// 	int num = bottom[0]->num();

// 	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();
// 	float margin = triplet_my_loss_param.margin();
// 	Dtype* dis_data = dis_.mutable_cpu_data();
// 	Dtype* mask_data = mask_.mutable_cpu_data();

// 	set_mask(bottom);
// 	Dtype loss = 0;
// 	int cnt = 0;

// 	for(int i = 0; i < num; i++){
// 		for(int j = i + 1; j < num; j++){
// 			for(int k = 0; k < num; k++){
// 				if(mask_data[i * num * num + j * num + k] == 1){
// 					Dtype tloss1 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[i * num + k] + Dtype(margin));
// 					Dtype tloss2 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[j * num + k] + Dtype(margin));
// 					loss += tloss1 + tloss2;
// 					// loss += tloss1;
// 					cnt ++;
// 				}
// 			}
// 		}
// 	}
// 	loss = loss / Dtype(cnt) / 2;
//     if (cnt == 0)
//         loss = Dtype(0.0);
// 	top[0]->mutable_cpu_data()[0] = loss;
// }

// template <typename Dtype>
// void TripletMyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

// 	const Dtype* bottom_data = bottom[0]->gpu_data();
// 	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
// 	int count = bottom[0]->count();
// 	int num = bottom[0]->num();
// 	int dim = bottom[0]->count() / bottom[0]->num();
// 	const Dtype alpha = top[0]->cpu_diff()[0];

// 	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();
// 	float margin = triplet_my_loss_param.margin();

// 	Dtype* dis_data = dis_.mutable_cpu_data();
// 	Dtype* mask_data = mask_.mutable_cpu_data();


// 	caffe_gpu_set(count, Dtype(0.0), bottom_diff);
// 	int cnt = 0;

// 	for(int i = 0; i < num; i++){
// 		for(int j = i + 1; j < num; j++){
// 			const Dtype* fori = bottom_data + i * dim;
// 		    const Dtype* fpos = bottom_data + j * dim;
// 		    Dtype* fori_diff = bottom_diff + i * dim;
// 			Dtype* fpos_diff = bottom_diff + j * dim;
// 			for(int k = 0; k < num; k++){
// 				if(mask_data[i * num * num + j * num + k] == 1){
// 					Dtype tloss1 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[i * num + k] + Dtype(margin));
// 					Dtype tloss2 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[j * num + k] + Dtype(margin));
					
// 					const Dtype* fneg = bottom_data + k * dim;
// 					Dtype* fneg_diff = bottom_diff + k * dim;
// 					if(tloss1 > 0){
// 						caffe_gpu_axpy(
// 							dim,
// 							-alpha,
// 							fpos,
// 							fori_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							alpha,
// 							fneg,
// 							fori_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							-alpha,
// 							fori,
// 							fpos_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							alpha,
// 							fpos,
// 							fpos_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							alpha,
// 							fori,
// 							fneg_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							-alpha,
// 							fneg,
// 							fneg_diff);
// 					}
// 					if(tloss2 > 0){
// 						caffe_gpu_axpy(
// 							dim,
// 							-alpha,
// 							fpos,
// 							fori_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							alpha,
// 							fori,
// 							fori_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							-alpha,
// 							fori,
// 							fpos_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							alpha,
// 							fneg,
// 							fpos_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							alpha,
// 							fpos,
// 							fneg_diff);
// 						caffe_gpu_axpy(
// 							dim,
// 							-alpha,
// 							fneg,
// 							fneg_diff);
// 					}
// 					cnt ++;
// 				}
// 			}
// 		}
// 	}
// 	// for(int i = 0; i < count; i ++){
// 	// 	bottom[0]->mutable_cpu_diff()[i] = bottom[0]->mutable_cpu_diff()[i] / cnt ;
// 	// 	if (cnt == 0)
// 	// 	bottom[0]->mutable_cpu_diff()[i] = Dtype(0.0);
// 	// }	
// 	if(cnt == 0)
// 		caffe_gpu_set(count, Dtype(0.0), bottom_diff);
// 	else
// 		Dtype beta = Dtype(1.0)/Dtype(cnt);
// 		caffe_gpu_scal(count, beta, bottom_diff);
// }

INSTANTIATE_LAYER_GPU_FUNCS(TripletMyLossLayer);

}  // namespace caffe