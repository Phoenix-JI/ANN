//
//  Header.h
//  Comp3046_Phase_3
//
//  Created by Phoenix JI .
//  Copyright © 2019 Phoenix JI. All rights reserved.
//

#ifndef ANN_h
#define ANN_h

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <time.h>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

class ANN{
    
public:
    
    float r;
    int epochs;
    int mbz;                        //mini_batch_size
    int layers;     // Number of layers
    
    vector <int> neu_eachlayer;  // The vector counts the number of neurons of each layer
    
    vector <vector<float> > V0;   // The layers distribution
    
    vector <vector<float> > Errors; // The 2D vector to store the all errors
    
    vector <vector<float> > Bias;  // The 2D vector to store the Bias
    
    vector <vector<vector<float> > > Weights;  // The 3D Vector to store weights of all layers
    
public:
    
    ANN();
    void set_LearningRate(float lr);
    void set_epochs(int epochs);
    void set_num_batch(int mbz);
    float sigmoid(float x);
    void Weights_Bias_Initilization(vector<float> &Z);
    
    void setLayer(int layers, vector<int> neu_eachlayer);
    void feedforward(vector<float> &Z);
    float sigmoid_Derivative(float y);
    
    void Output_error(vector<float> &Actual);
    void Backpropagate_error();
    void train(vector<vector<float> > &X,vector<vector<float> > &Actual );
    float loss(vector<float> &X, vector<float> &Y);
    int inference (vector<float> &Z);
    void Bias_load();
    void Bias_store();
    void Weights_load();
    void Weights_store();
    
};


#endif /* Header_h */
