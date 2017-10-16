#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

class normal_distribution {
public:
  double mean;
  double variance;

  normal_distribution(double mean, double variance);
  double probability_density(double value);
};

class GNB {
public:

	vector<string> possible_labels = {"left","keep","right"};


 	GNB();
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);
  string predict(vector<double>);
  std::map<string, vector<normal_distribution>> all_distributions;
};


#endif


