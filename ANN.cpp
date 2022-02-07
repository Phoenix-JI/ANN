//
//  ANN.cpp
//  Comp3046_Phase_3
//
//  Created by Phoenix Constantine on 5/5/19.
//  Copyright © 2019 Phoenix JI. All rights reserved.
//

#include "ANN.h"
using namespace std;

ANN::ANN(){
    
    
    
}

void  ANN::set_LearningRate(float lr){
    this->r=lr;
    
}

void  ANN::set_epochs(int epochs){
    this->epochs=epochs;
}

void  ANN::set_num_batch(int mbz){
    this->mbz=mbz;
}


float ANN::sigmoid(float x){
    return 1 / (1 + exp(-x));
}

float ANN::sigmoid_Derivative(float y){
    
    return y*(1-y);
}


float ANN:: loss(vector<float> &X, vector<float> &Y){
    
    float loss=0;
    
    for(int i=0;i<X.size();i++){
        loss=loss+pow((Y[i]-X[i]),2);
    }
    
    loss=loss*0.5;
    
    return loss;
    
}


void ANN::setLayer(int layers, vector <int> neu_eachlayer){
    
    this->layers=layers;
    this->neu_eachlayer=neu_eachlayer;
    
    vector<vector<float> > V1;
    
    for (int i=0;i<layers;i++){
        
        V1.push_back(vector<float>());
        
        for(int j=0; j<neu_eachlayer[i]; j++){
            
            V1[i].push_back(0);
        }
        
    }
    V0=V1;
    
}

void ANN::Weights_Bias_Initilization(vector<float> &Z){
    
    srand (static_cast <unsigned> (time(0)));
    
    for(int q=0;q<V0.size();q++){
        Bias.push_back(vector<float>());
        
        
        for(int i=0;i<V0[q].size();i++){             //Set random bias for all the  layers
            float bias = (float) rand()/RAND_MAX-0.5;
            Bias[q].push_back(bias); //
        }
    }
    
    
    for(int q=0;q<V0.size();q++){
        
        Weights.push_back(vector<vector<float> >());
        
        for(int i=0;i<V0[q].size();i++){
            Weights[q].push_back(vector<float>());
            
            if(q==0){
                
                for(int j=0;j<Z.size();j++){
                    float iWeight = (float) rand()/RAND_MAX-0.5;         // Set random weights  for the first layer
                    Weights[0][i].push_back(iWeight);
                }
                
            }else{
                
                for(int j=0;j<V0[q-1].size();j++){
                    float iWeight = (float) rand()/RAND_MAX-0.5;   // Set random weights  for the hidden layers
                    Weights[q][i].push_back(iWeight);
                }
                
            }
        }
    }
    
}


void  ANN::feedforward( vector<float> &Z){
    
    setLayer(layers, neu_eachlayer);
    
    for(int i=0;i<V0.size();i++){

#  pragma omp parallel for num_threads(4)

        for(int j=0;j<V0[i].size();j++){
            
            float sum=0;
            
            if(i==0){
                
                for(int q=0;q<Z.size();q++){
                    
                    sum=sum+Z[q]*Weights[0][j][q];
                    
                }
                
            } else {
                
                for(int q=0;q<V0[i-1].size();q++){
                    
                    sum=sum+V0[i-1][q]*Weights[i][j][q];
                    
                }
            }
            
            V0[i][j]=sigmoid(sum+Bias[i][j]);
            
        }
        
    }
    
}


void ANN::Output_error(vector<float> &Actual){
    
    vector<vector<float> > tempError;
    
    for(int i=0;i<layers;i++){
        tempError.push_back(vector<float>());
    }
    
    for(int i=0;i<V0[layers-1].size();i++){
        float err=0;
        
        err=(V0[layers-1][i]-Actual[i])*sigmoid_Derivative(V0[layers-1][i]);
        
        tempError[layers-1].push_back(err);
    }
    
    Errors=tempError;
    
    
    
    
}


void ANN:: Backpropagate_error(){
    
    for(int i=layers-2;i>=0;i--){
        
        for(int j=0;j<V0[i].size();j++){
            float err=0;

#  pragma omp parallel for num_threads(4)

            for(int q=0;q<V0[i+1].size();q++){
                
                err=err+Weights[i+1][q][j]*Errors[layers-1][q];
                
            }
            
            Errors[i].push_back(err*sigmoid_Derivative(V0[i][j]));
        }
        
    }
    
}


void ANN:: train(vector<vector<float> > &X,vector<vector<float> > &Actual ){
    
    unsigned seed = chrono::system_clock::now ().time_since_epoch ().count ();
    
    shuffle (X.begin(), X.end(), default_random_engine (seed));
    shuffle (Actual.begin(), Actual.end(), default_random_engine (seed));
    
    
    for(int ep=0;ep<epochs;ep++){
        
        clock_t  start, end;
        start  = clock();
        
        vector<vector<float> > Grad_B;
        vector<vector<vector<float> > > Grad_W;
        
        for(int p=0;p<Bias.size();p++){
            
            Grad_B.push_back(vector<float>());
            
            for(int i=0;i<Bias[p].size();i++){
                
                Grad_B[p].push_back(0);
            }
        }
        
        
        for(int p=0;p<Weights.size();p++){
            
            Grad_W.push_back(vector<vector<float> >());
            
            for(int i=0;i<Weights[p].size();i++){
                
                Grad_W[p].push_back(vector<float>());
                
                if(p==0){
                    for(int j=0;j<X[0].size();j++){
                        
                        Grad_W[0][i].push_back(0);
                    }
                    
                }else{
                    
                    for(int j=0;j<Weights[p-1].size();j++){
                        
                        Grad_W[p][i].push_back(0);
                        
                    }
                }
            }
        }
        
        float loss_t=0;
        
        int amount=X.size()/mbz; // 4000=40000/10
        
        vector<int> batch;
        
        int n=0;
        for(int i=0;i<=mbz;i++){ //i<=10
            batch.push_back(n); // batch 0 4000 8000 12000 16000 20000 24000 28000 32000 36000 40000 40001*
            n=n+amount;
        }
        batch.push_back(X.size());
        
        
        for(int m=0;m<batch.size()-1;m++){ //m=0 1 2 3 4 5 6 7 8 9 10
            
            for(int q=batch[m];q<batch[m+1];q++){
                
                feedforward(X[q]);
                Output_error(Actual[q]);
                Backpropagate_error();
                
                if(q==0){
                    
                    cout<<"Predict && label"<<endl;
                    
                    for(int i=0;i<V0[layers-1].size();i++){
                        cout<<V0[layers-1][i]<<" ";
                    }
                    cout<<endl;
                    
                    for(int i=0;i<V0[layers-1].size();i++){
                        cout<<Actual[q][i]<<" ";
                    }
                    cout<<endl;
                    
                    
                }
                
                for(int l=layers-1;l>=0;l--){
                    
                    for(int d=0;d<Errors[l].size();d++){
                        
                        Grad_B[l][d]= Grad_B[l][d]+Errors[l][d];
                        
                        if(l==0){
#  pragma omp parallel for num_threads(4)
                            for(int e=0;e<X[0].size();e++){
                                
                                Grad_W[l][d][e]=Grad_W[l][d][e]+Errors[l][d]*X[q][e];
                                
                            }
                            
                        }else{
#  pragma omp parallel for num_threads(4)
                            for(int e=0;e<V0[l-1].size();e++){
                                
                                Grad_W[l][d][e]=Grad_W[l][d][e]+Errors[l][d]*V0[l-1][e];
                                
                            }
                        }
                        
                    }
                    
                }
                
                
                loss_t=loss_t+loss(V0[layers-1], Actual[q]);
            }
            
            
            
            for(int i=0;i<V0.size();i++){
                for(int j=0;j<V0[i].size();j++){
                    
                    Bias[i][j]=Bias[i][j]+Grad_B[i][j]*(-r)/amount;
                    
                    for(int q=0;q<Weights[i][j].size();q++){
                        
                        Weights[i][j][q]=Weights[i][j][q]+Grad_W[i][j][q]*(-r)/amount;
                        
                    }
                    
                }
                
            }
            
            
            
        }
        end = clock();
        cout<<endl;
        cout<<"Epoch "<<ep<<" finished"<<"  Loss value: "<<loss_t/X.size();
        
        cout<<"  Time needed:  "<<(double)(end-start)/CLOCKS_PER_SEC * 1000<<" milliseconds";
        cout<<endl;
        
    }
    
    
}



int ANN:: inference(vector<float> &X){
    
    feedforward(X);
    
    for(int i=0;i<V0[V0.size()-1].size();i++){
        cout<<V0[V0.size()-1][i]<<"^^^ ";
    }
    
    vector<float>::iterator it;
    it=max_element(V0[V0.size()-1].begin(), V0[V0.size()-1].end());
    
    cout<<endl;
    cout<<*it<<endl;
    
    cout<<endl;
    
    int d=-1;
    for(int i=0;i<10;i++){
        
        while(V0[V0.size()-1][i]==*it){
            d=i;
            break;
            
        }
    }
    
    cout<<d<<endl;
    
    return d;
    
}


void ANN::Bias_load(){
    char filename[80];
    cout << "Load (Bias) file name：";
    cin >> filename;
    strcat(filename, ".txt");
    
    ifstream inFile;
    inFile.open(filename);
    
    for(int i=0;i<Bias.size();i++){
        for(int j=0;j<Bias[i].size();j++){
            inFile >> Bias[i][j];
        }
    }
    
    for(int i=0;i<Bias.size();i++){
        for(int j=0;j<Bias[i].size();j++){
            cout<< Bias[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    inFile.close();
}

void ANN::Bias_store(){
    char filename[80];
    cout << "Store (Bias) file name：";
    cin >> filename;
    strcat(filename, ".txt");
    
    ofstream outFile;
    outFile.open(filename);
    
    for(int i=0;i<Bias.size();i++){
        for(int j=0;j<Bias[i].size();j++){
            outFile <<Bias[i][j]<<" " ;
        }
        outFile<<endl;
    }
    outFile.close();
}

void ANN::Weights_load(){
    char filename[80];
    cout << "Load (Weights) file name：";
    cin >> filename;
    strcat(filename, ".txt");
    
    ifstream inFile;
    inFile.open(filename);
    
    for(int i=0;i<Weights.size();i++){
        for(int j=0;j<Weights[i].size();j++){
            for(int k=0;k<Weights[i][j].size();k++){
                inFile >> Weights[i][j][k];
            }
        }
    }
    
    inFile.close();
}

void ANN:: Weights_store(){
    
    char filename[80];
    cout << "Store (Weights) file name：";
    cin >> filename;
    strcat(filename, ".txt");
    
    ofstream outFile;
    outFile.open(filename);
    
    for(int i=0;i<Weights.size();i++){
        for(int j=0;j<Weights[i].size();j++){
            for(int k=0;k<Weights[i][j].size();k++){
                outFile <<Weights[i][j][k]<<" " ;
            }
            outFile<<endl;
        }
        outFile<<endl;
    }
    outFile<<endl;
    outFile.close();
    
}
