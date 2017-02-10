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
void TripletMyLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	diff_ap_.Reshape(1, bottom[0]->channels(), 1, 1);
	diff_an_.Reshape(1, bottom[0]->channels(), 1, 1);

	dis_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
	mask_.Reshape(bottom[0]->num(), bottom[0]->num(), bottom[0]->num(), 1);
}

template <typename Dtype>
void TripletMyLossLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom){
	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();
	int hard_num = triplet_my_loss_param.hard_num();
	int rand_num = triplet_my_loss_param.rand_num();
	float margin = triplet_my_loss_param.margin();
	int bg_label = triplet_my_loss_param.bg_label();
	int neg_num = hard_num + rand_num;

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	caffe_set(dis_.count(), Dtype(0.0), dis_data);
	caffe_set(mask_.count(), Dtype(0.0), mask_data);
	for(int i = 0; i < num; i ++){
		for(int j = i + 1; j < num; j ++){
			caffe_sub(dim,
					bottom_data + i*dim,
					bottom_data + j*dim,
					diff_ap_.mutable_cpu_data());
			Dtype ts = caffe_cpu_dot(dim,
								diff_ap_.cpu_data(),
								diff_ap_.cpu_data());
			dis_data[i * num + j] = ts;
			dis_data[j * num + i] = ts;
		}
	}
	//select samples
	vector<pair<float, int> >negpairs;
	vector<int> sid1;
	vector<int> sid2;
	for(int i = 0; i < num; i++){
		if(static_cast<int>(label[i]) == bg_label)
			continue;
		for(int j = i + 1; j < num; j++){
			if(label[i] == label[j]){
				negpairs.clear();
				sid1.clear();
				sid2.clear();
				for(int k = 0; k < num; k++){
					if(static_cast<int>(label[k]) == bg_label)
						continue;
					if(label[k] != label[i]){
						Dtype tloss = max(Dtype(0.0), dis_data[i * num + j] - dis_data[i * num + k] + Dtype(margin));
						if(tloss != 0) 
							negpairs.push_back(make_pair(dis_data[i * num + k], k));
					}
				}
				if(negpairs.size() <= neg_num){
					for(int k = 0; k < negpairs.size(); k++){
						int id = negpairs[k].second;
						mask_data[i * num * num + j * num + id] = 1;
					}
				}
				else{
					sort(negpairs.begin(), negpairs.end());

					for(int k = 0; k < neg_num; k++){
						sid1.push_back(negpairs[k].second);
					}
					for(int k = neg_num; k < negpairs.size(); k++){
						sid2.push_back(negpairs[k].second);
					}
					std::random_shuffle(sid1.begin(), sid1.end(), myrandom);
					for(int k = 0; k < min(hard_num, (int)(sid1.size()) ); k++){
						mask_data[i * num * num + j * num + sid1[k]] = 1;
					}
					for(int k = hard_num; k < sid1.size(); k ++){
						sid2.push_back(sid1[k]);
					}
					std::random_shuffle(sid2.begin(), sid2.end(), myrandom);
					for(int k = 0; k < min(rand_num, (int)(sid2.size()) ); k++){
						mask_data[i * num * num + j * num + sid2[k]] = 1;
					}
				}
			}
		}
	}
}

template <typename Dtype>
void TripletMyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	int num = bottom[0]->num();

	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();
	float margin = triplet_my_loss_param.margin();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	set_mask(bottom);
	Dtype loss = 0;
	int cnt = 0;

	for(int i = 0; i < num; i++){
		for(int j = i + 1; j < num; j++){
			for(int k = 0; k < num; k++){
				if(mask_data[i * num * num + j * num + k] == 1){
					Dtype tloss1 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[i * num + k] + Dtype(margin));
					Dtype tloss2 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[j * num + k] + Dtype(margin));
					loss += tloss1 + tloss2;
					// loss += tloss1;
					cnt ++;
				}
			}
		}
	}
    if (cnt == 0)
        loss = Dtype(0.0);
    else
    	loss = loss / Dtype(cnt) / 2;
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletMyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	const Dtype alpha = top[0]->cpu_diff()[0];

	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();
	float margin = triplet_my_loss_param.margin();

	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();


	caffe_set(count, Dtype(0.0), bottom_diff);
	int cnt = 0;

	for(int i = 0; i < num; i++){
		for(int j = i + 1; j < num; j++){
			const Dtype* fori = bottom_data + i * dim;
		    const Dtype* fpos = bottom_data + j * dim;
		    Dtype* fori_diff = bottom_diff + i * dim;
			Dtype* fpos_diff = bottom_diff + j * dim;
			for(int k = 0; k < num; k++){
				if(mask_data[i * num * num + j * num + k] == 1){
					Dtype tloss1 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[i * num + k] + Dtype(margin));
					Dtype tloss2 = max(Dtype(0.0), dis_data[i * num + j] - dis_data[j * num + k] + Dtype(margin));
					
					const Dtype* fneg = bottom_data + k * dim;
					Dtype* fneg_diff = bottom_diff + k * dim;
					if(tloss1 > 0){
						caffe_axpy(
							dim,
							-alpha,
							fpos,
							fori_diff);
						caffe_axpy(
							dim,
							alpha,
							fneg,
							fori_diff);
						caffe_axpy(
							dim,
							-alpha,
							fori,
							fpos_diff);
						caffe_axpy(
							dim,
							alpha,
							fpos,
							fpos_diff);
						caffe_axpy(
							dim,
							alpha,
							fori,
							fneg_diff);
						caffe_axpy(
							dim,
							-alpha,
							fneg,
							fneg_diff);
					}
					if(tloss2 > 0){
						caffe_axpy(
							dim,
							-alpha,
							fpos,
							fori_diff);
						caffe_axpy(
							dim,
							alpha,
							fori,
							fori_diff);
						caffe_axpy(
							dim,
							-alpha,
							fori,
							fpos_diff);
						caffe_axpy(
							dim,
							alpha,
							fneg,
							fpos_diff);
						caffe_axpy(
							dim,
							alpha,
							fpos,
							fneg_diff);
						caffe_axpy(
							dim,
							-alpha,
							fneg,
							fneg_diff);
					}
					cnt ++;
				}
			}
		}
	}
	// for(int i = 0; i < count; i ++){
	// 	bottom_diff[i] = bottom_diff[i] / cnt ;
	// 	if (cnt == 0)
	// 		bottom_diff[i] = Dtype(0.0);
	// }
	if (cnt == 0)
		caffe_set(count, Dtype(0.0), bottom_diff);
	else{
		Dtype beta = Dtype(1.0 / cnt);
		caffe_scal(count, beta, bottom_diff);
	}
}

// template <typename Dtype>
// void TripletMyLossLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom){
// 	const Dtype* bottom_label = bottom[1]->cpu_data();
// 	const Dtype* bottom_data = bottom[0]->cpu_data();
// 	const int num = bottom[0]->num();
// 	const int channels = bottom[0]->channels();
// 	Dtype* mask_data = mask_.mutable_cpu_data();
// 	Dtype* dis_data = dis_.mutable_cpu_data();
// 	caffe_set(mask_.count(), Dtype(0.0), mask_data);

// 	map<int, vector<int> > label_data_map;
// 	int max_label = 0;

// 	for(int i = 0; i < num; i++) {
// 		const int label_value = static_cast<int>(bottom_label[i]);
// 		if(label_value > max_label)
// 				max_label = label_value;
// 		if(label_data_map.count(label_value) > 0){
// 			label_data_map[label_value].push_back(i);
// 		}else{
// 			vector<int> tmp;
// 			tmp.push_back(i);
// 			label_data_map[label_value] = tmp;
// 		}
// 	}

// 	for(int i = 0 ; i < num; i++){
// 		const int label_value = static_cast<int>(bottom_label[i]);
		
// 		int positive_index = i;
// 		if(label_data_map[label_value].size() != 1){
// 			while(positive_index == i)
// 				positive_index = label_data_map[label_value][rand() % label_data_map[label_value].size()];
// 		}
		
// 		int negative_label = label_value;
// 		while(negative_label == label_value || label_data_map.count(negative_label) == 0){
// 			negative_label = rand() % (max_label + 1);
// 		}
// 		int negative_index = label_data_map[negative_label][rand() % label_data_map[negative_label].size()];
// 		mask_data[i * 3] = i;
// 		mask_data[i * 3 + 1] = positive_index;
// 		mask_data[i * 3 + 2] = negative_index;
// 	}

// 	for(int i = 0; i < num; i ++){
// 		for(int j = i + 1; j < num; j ++){
// 			caffe_sub(channels,
// 					bottom_data + i*channels,
// 					bottom_data + j*channels,
// 					diff_ap_.mutable_cpu_data());
// 			Dtype ts = caffe_cpu_dot(channels,
// 								diff_ap_.cpu_data(),
// 								diff_ap_.cpu_data());
// 			dis_data[i * num + j] = ts;
// 			dis_data[j * num + i] = ts;
// 		}
// 	}
// }

// template <typename Dtype>
// void TripletMyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {

// 	int num = bottom[0]->num();
// 	const int channels = bottom[0]->channels();

// 	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();
// 	float margin = triplet_my_loss_param.margin();
// 	Dtype* mask_data = mask_.mutable_cpu_data();

// 	set_mask(bottom);
// 	Dtype loss = 0;
// 	for(int i = 0; i < num; i++){
// 		const Dtype* anchor = bottom[0]->cpu_data() + (static_cast<int>(mask_data[i * 3]) * channels);
// 		const Dtype* positive = bottom[0]->cpu_data() + (static_cast<int>(mask_data[i * 3 + 1]) * channels);
// 		const Dtype* negative = bottom[0]->cpu_data() + (static_cast<int>(mask_data[i * 3 + 2]) * channels);
// 		caffe_sub(
// 			channels, 
// 			anchor,
// 			positive,
// 			diff_ap_.mutable_cpu_data());
// 		caffe_sub(
// 			channels, 
// 			anchor,
// 			negative,
// 			diff_an_.mutable_cpu_data());
// 		Dtype dist_sq_ap_ = caffe_cpu_dot(
// 								channels, 
// 								diff_ap_.cpu_data(),
// 								diff_ap_.cpu_data());
// 		Dtype dist_sq_an_ = caffe_cpu_dot(
// 								channels, 
// 								diff_an_.cpu_data(),
// 								diff_an_.cpu_data());
// 		Dtype dist = max(Dtype(0), dist_sq_ap_ - dist_sq_an_ + Dtype(margin));
// 		loss += dist;
// 		// cout << "dist = " << dist << "\n";
// 		if(dist == 0){
// 			mask_data[i * 3] = 0;
// 			mask_data[i * 3 + 1] = 0;
// 			mask_data[i * 3 + 2] = 0;
// 		}
// 	}
// 	loss = loss / num / Dtype(2);
// 	top[0]->mutable_cpu_data()[0] = loss;
// }

// template <typename Dtype>
// void TripletMyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

// 	const Dtype* bottom_data = bottom[0]->cpu_data();
// 	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
// 	int count = bottom[0]->count();
// 	int num = bottom[0]->num();
// 	const int channels = bottom[0]->channels();
// 	const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[0]->num());

// 	TripletMyLossParameter triplet_my_loss_param = this->layer_param_.triplet_my_loss_param();

// 	Dtype* mask_data = mask_.mutable_cpu_data();


// 	caffe_set(count, Dtype(0.0), bottom_diff);

// 	for(int i = 0; i < num; i++){
// 		if(mask_data[i * 3] == 0 && mask_data[i * 3] == 0 && mask_data[i * 3] == 0)
// 			continue;

// 		const Dtype* fori = bottom_data + static_cast<int>(mask_data[i * 3]) * channels;
// 		const Dtype* fpos = bottom_data + static_cast<int>(mask_data[i * 3 + 1]) * channels;
// 		const Dtype* fneg = bottom_data + static_cast<int>(mask_data[i * 3 + 2]) * channels;
// 		Dtype* fori_diff = bottom_diff + static_cast<int>(mask_data[i * 3]) * channels;
// 		Dtype* fpos_diff = bottom_diff + static_cast<int>(mask_data[i * 3 + 1]) * channels;
// 		Dtype* fneg_diff = bottom_diff + static_cast<int>(mask_data[i * 3 + 2]) * channels;
// 		caffe_cpu_axpby(
// 			channels,
// 			-alpha,
// 			fpos,
// 			Dtype(1.0),
// 			fori_diff);
// 		caffe_cpu_axpby(
// 			channels,
// 			alpha,
// 			fneg,
// 			Dtype(1.0),
// 			fori_diff);
// 		caffe_cpu_axpby(
// 			channels,
// 			-alpha,
// 			fori,
// 			Dtype(1.0),
// 			fpos_diff);
// 		caffe_cpu_axpby(
// 			channels,
// 			alpha,
// 			fpos,
// 			Dtype(1.0),
// 			fpos_diff);
// 		caffe_cpu_axpby(
// 			channels,
// 			alpha,
// 			fori,
// 			Dtype(1.0),
// 			fneg_diff);
// 		caffe_cpu_axpby(
// 			channels,
// 			-alpha,
// 			fneg,
// 			Dtype(1.0),
// 			fneg_diff);
// 	}
// }

#ifdef CPU_ONLY
STUB_GPU(TripletMyLossLayer);
#endif

INSTANTIATE_CLASS(TripletMyLossLayer);
REGISTER_LAYER_CLASS(TripletMyLoss);

}  // namespace caffe
