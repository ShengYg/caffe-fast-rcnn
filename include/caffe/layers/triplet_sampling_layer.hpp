#ifndef CAFFE_TRIPLET_SAMPLING_LAYER_HPP_
#define CAFFE_TRIPLET_SAMPLING_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class TripletSamplingLayer : public Layer<Dtype> {
 public:
  explicit TripletSamplingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   /**
  *Unlike most loss layers, in the TripletLossLayer we can backpropagate to the first three inputs.
  */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index!= 1; // the last bottom blob not need.
  }

  virtual inline const char* type() const { return "TripletSampling"; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 Blob<Dtype> top0_map;
 Blob<Dtype> top1_map;
 Blob<Dtype> top2_map;
};
}

#endif