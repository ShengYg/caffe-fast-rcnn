#ifndef CAFFE_TRIPLET_MY_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_MY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

static int myrandom (int i) { return caffe_rng_rand()%i;}

template <typename Dtype>
class TripletMyLossLayer : public LossLayer<Dtype> {
 public:
  explicit TripletMyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletMyLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void set_mask(const vector<Blob<Dtype>*>& bottom);
  void set_mask_gpu(const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> mask_;
  Blob<Dtype> dis_;
  Blob<Dtype> diff_ap_;
  Blob<Dtype> diff_an_;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_MY_LOSS_LAYER_HPP_