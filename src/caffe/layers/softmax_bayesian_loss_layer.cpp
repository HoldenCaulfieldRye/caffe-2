#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithBayesianLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  labels_.Reshape(bottom[1]->num(), 1, 1, 1);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype SoftmaxWithBayesianLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  // what is _prob ? looks like instantiation of some class
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;

  float prior[2];
  prior[0] = 0;
  prior[1] = 0;  
  for (int i = 0; i < num; ++i)
    prior[static_cast<int>(label[i])] += 1.0 / num;
  
  Dtype loss = 0;
  // std::cout << "loss: ";
  for (int i = 0; i < num; ++i) {
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])],
		     Dtype(FLT_MIN))) / (dim*num*prior[static_cast<int>(label[i])]);
    // std::cout << loss << ", ";
  }
  // std::cout << std::endl;
  return loss;
}

template <typename Dtype>
// computes dE/dz for every neuron input vector z = <x,w>+b
// this does NOT update the weights, it merely calculates dy/dz
void SoftmaxWithBayesianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();         //batchSize, num imgs
  int dim = prob_.count() / num; //num neurons, dimensionality
  
  float prior[2] = {0,0};
  for (int i = 0; i < num; ++i)
    prior[static_cast<int>(label[i])] += 1.0 / num;

  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1 ;
  }
  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  
  for (int j = 0; j < dim; ++j) {
    for (int i = 0; i < num; ++i)
      bottom_diff[i * dim + j] /= (static_cast<float>(prior[static_cast<int>(label[i])])*dim);
  }
}
/*
template <typename Dtype>
void SoftmaxWithBayesianLossLayer<Dtype>::Backward_cpu_old(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
  }
  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
}
*/

INSTANTIATE_CLASS(SoftmaxWithBayesianLossLayer);


}  // namespace caffe
