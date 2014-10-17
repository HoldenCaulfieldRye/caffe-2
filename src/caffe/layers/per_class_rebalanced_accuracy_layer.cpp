// Copyright 2014 BVLC and contributors.
// #include <iostream>
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cfloat>
// #include <iostream>
// #include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {
  
template <typename Dtype>
void PerClassRebalancedAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  int dim = bottom[0]->count() / bottom[0]->num();
  int label_count = bottom[1]->count();
  CHECK_EQ(dim, label_count)
      << "Oh shit I thought dim and label_count were equal";
  (*top)[0]->Reshape(1, 1, 1, dim);
  accuracies_.ReshapeLike((*top)[0]);
  labels_count_.Reshape(1, 1, 1, dim);  
}


template <typename Dtype>
void PerClassRebalancedAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data(); //threshold_layer calls this bottom_data
  // Dtype accuracy = 0;
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  Dtype* labels_count = labels_count_.mutable_cpu_data();
  Dtype* accuracies = accuracies_.mutable_cpu_data();
  caffe_set(labels_count_.count(), Dtype(FLT_MIN), labels_count);
  caffe_set(accuracies_.count(), Dtype(FLT_MIN), accuracies);
  
  for (int i = 0; i < num; ++i) {
    //count freq of each class
    labels_count[static_cast<int>(bottom_label[i])] += 1.0;
    //determine whether correctly classified
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      //find which class gets highest prob
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i]))
      accuracies[static_cast<int>(bottom_label[i])] += 1.0;
  }

  for (int j = 0; j < dim; ++j) 
    accuracies[j] /= static_cast<float>(labels_count[j]);
      
  // LOG(INFO) << "Accuracies, class by class: " << accuracy;
  //can I do this or does 
  caffe_copy(dim, accuracies, (*top)[0]->mutable_cpu_data());
  // (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  // (*top)[0]->mutable_cpu_data()[0] = 1;
  // (*top)[0]->mutable_cpu_data()[0] = 2;
  // (*top)[0]->mutable_cpu_data()[0] = 3;
  // Accuracy layer should not be used as a loss function.
} 

INSTANTIATE_CLASS(PerClassRebalancedAccuracyLayer);

}  // namespace caffe
