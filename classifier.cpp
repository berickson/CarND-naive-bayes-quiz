#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

template<typename T> string to_string(vector<T> v){
    stringstream ss;
    ss << "{";
    for(int i = 0; i < v.size() -1; i++) {
        ss << v[i] << ", ";
    }
    ss << v[v.size()-1] << "}";
    return ss.str();
}

normal_distribution::normal_distribution(double mean, double variance) {
  this->mean = mean;
  this->variance = variance;
}

double normal_distribution::probability_density(double value) {
  double d = value-mean;
  return 1/sqrt(2*M_PI*variance)*exp(-(d*d)/(2*variance));
}

vector<double> features_for_data(vector<double> data) {
  double lane_width = 4;
  double s = data[0];
  double d = data[1];
  double s_dot = data[2];
  double d_dot = data[3];
  return vector<double>{s, fmod(d, lane_width), s_dot, d_dot};
}


void GNB::train(vector<vector<double>> data, vector<string> labels)
{
    std::map<string, vector<vector<double>>> label_to_data;
    std::map<string, vector<double>> all_m0;
    std::map<string, vector<double>> all_m1;
    std::map<string, vector<double>> all_m2;
    for(auto label:possible_labels) {
        all_m0[label] = features_for_data({0,0,0,0});
        all_m1[label] = features_for_data({0,0,0,0});
        all_m2[label] = features_for_data({0,0,0,0});
    }
    
    
    for(int i = 0; i < labels.size() ; i++) {
        auto label = labels[i];
        auto state = data[i];
        //cout << labels[i] << endl;
        label_to_data[label].push_back(state);
        
        for(int j = 0; j < state.size(); j++) {
            double d = state[j];
            all_m0[label][j] += 1;
            all_m1[label][j] += d;
            all_m2[label][j] += d*d;
        }
    }
    cout << "#########################" << endl;

    for(auto label:possible_labels) {
      all_distributions[label]={};
      for(int i = 0; i < 4; i++) {
        double m0 = all_m0[label][i];
        double m1 = all_m1[label][i];
        double m2 = all_m2[label][i];
        double mean = m1/m0;
        double variance = m2/m0-mean*mean;
        double std = sqrt(variance);

        cout << label << endl;
        cout << "------------" << endl;
        cout << "m0: " << to_string(m0) << endl;
        cout << "m1: " << to_string(m1) << endl;
        cout << "m2: " << to_string(m2) << endl;
        cout << "mean: " << to_string(mean) << endl;
        cout << "variance: " << to_string(variance) << endl;
        cout << "std: " << to_string(std) << endl;
        cout << endl;
        all_distributions[label].push_back({mean, variance});
      }
      //print_vector(m1[label]);
    }
    //cout << m1 << endl;
    


	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/

  string best_label = "";
  double best_odds = 0.0;
  vector<double> features = features_for_data(sample);
  for(string label : possible_labels) {
    double odds = 1.0;
    for(int d = 0; d < features.size(); d++ ) {
      odds *= all_distributions[label][d].probability_density(features[d]);
    }
    if(odds > best_odds){
      best_odds = odds;
      best_label = label;
    }
  }

  cout << "predicted:" << best_label << endl;

  return best_label;

}
