//
//  main_2.cpp
//  Comp3046_Phase_3
//
//  Created by Phoenix JI .
//  Copyright © 2019 Phoenix JI. All rights reserved.
//
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <time.h>
#include <algorithm>
//#include <omp.h>
#include "ANN.h"


using namespace std;

int main() {
    
    /* ANN Initilization */
    
    cout<<endl;
    int layers;
    cout<<"Input layer has been assigned "<<endl;
    cout<<"Please input the number of other layers: ";
    cin >> layers;
    cout<<endl;
    
    vector <int> neu_eachlayer ;
    cout<<endl;
    
    cout<<"Input layer is layer 1 "<<endl;
    for(int i=1;i<=layers;i++){
        
        if(i==layers){
            cout<<"Please input number of neurons of output layer (must be 10): ";
            int numl;
            cin >>numl;
            neu_eachlayer.push_back(numl);
            break;
        }else{
            cout<<"Please input number of neurons of layer "<<i+1<<": ";
        }
        
        int num;
        cin >>num;
        neu_eachlayer.push_back(num);
        
    }
    
    cout<<endl;
    
    /* Choose service*/
    
    int pe;
    cout<<"Please choose: "<<endl;
    cout<<endl;
    cout<<"1. Train ANN     2. Test ANN "<<endl;
    cout<<endl;
    cin >> pe;
    
    switch(pe){
        case 1 :
        {
            clock_t  start, end;
            start  = clock();
            
            vector< vector<float> > X_train;
            vector<float> y_train;
            
            ifstream myfile("train.txt");
            
            if (myfile.is_open())
            {
                cout << "Loading data ...\n";
                string line;
                while (getline(myfile, line))
                {
                    int x, y;
                    vector<float> X;
                    stringstream ss(line);
                    ss >> y;
                    y_train.push_back(y);
                    for (int i = 0; i < 28 * 28; i++) {
                        ss >> x;
                        X.push_back(x/255.0);
                    }
                    X_train.push_back(X);
                }
                
                myfile.close();
                cout << "Loading data finished.\n";
            }
            else
                cout << "Unable to open file" << '\n';
            
            
            vector<vector<float> > Actual;
            
            for(int i=0;i<y_train.size();i++){
                
                Actual.push_back(vector<float>());
                
                for (int j=0;j<10;j++){
                    
                    if(j==y_train[i]){
                        Actual[i].push_back(1);
                    }else {
                        Actual[i].push_back(0);
                    }
                    
                }
                
            }
            cout<<endl;
            
            int epochs;
            cout<<"Please input the epochs: ";
            cin >> epochs;
            cout<<endl;
            
            float lr;
            cout<<"Please input the learning rate: ";
            cin >> lr;
            cout<<endl;
            
            int mbz;
            cout<<"Please input the number of mini-batches: ";
            cin >> mbz;
            cout<<endl;
            
            ANN A1;
            A1.setLayer(layers,neu_eachlayer);
            A1.set_LearningRate(lr);
            A1.set_epochs(epochs);
            A1.Weights_Bias_Initilization(X_train[0]);
            A1.set_num_batch(mbz);
            A1.train(X_train,Actual);
            
            end = clock();
            cout<<"The time was:  "<< (end - start) * 1000 << " milliseconds" <<endl;
            cout<<endl;
            
            A1.Weights_store();
            A1.Bias_store();
            break;
        }
            
        case 2 :
        {
            /* Test Program */
            vector< vector<float> > X_test;
            vector<float> y_test;
            
            ifstream mytestfile("test.txt");
            
            if (mytestfile.is_open())
            {
                cout << "Loading data ...\n";
                string line;
                while (getline(mytestfile, line))
                {
                    int x, y;
                    vector<float> X;
                    stringstream ss(line);
                    ss >> y;
                    y_test.push_back(y);
                    for (int i = 0; i < 28 * 28; i++) {
                        ss >> x;
                        X.push_back(x/255.0);
                    }
                    X_test.push_back(X);
                }
                
                mytestfile.close();
                cout << "Loading data finished.\n";
            }
            else
                cout << "Unable to open file" << '\n';
            cout<<endl;
            
            ANN A2;
            
            A2.setLayer(layers,neu_eachlayer);
            A2.Weights_Bias_Initilization(X_test[0]);
            A2.Weights_load();
            A2.Bias_load();
            
            int acc=0;
            for(int i=0;i<X_test.size();i++){
                
                if(A2.inference(X_test[i])==y_test[i]){
                    acc++;
                }
            }
            
            cout<<"Accuracy: "<<(float) acc/y_test.size() <<endl;
            break;
        }
            
        default:
        {
            cout<<"Invalid Operation"<<endl;
        }
    }
  
  
    

 return 0;
    
}
