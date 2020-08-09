#include <bits/stdc++.h>
using namespace std;

#define PI (2*acos(0.0) )
int _count = 0, classCount, featureCount, sampleCount, class_1_count, class_2_count;
vector<double> fea_1_1, fea_2_1,fea_1_2, fea_2_2;
vector<int> classIndex_1, classIndex_2;
double fea_1_1_sum,fea_2_1_sum,fea_1_2_sum,fea_2_2_sum ,fea_1_mean[2],fea_2_mean[2],
        fea_1_sd[2], fea_2_sd[2], classMean[2];

double correct, totalTest;
double accuracy;
double getSd(vector<double> fea, double mean){
    double sum = 0;
    for(int i=0;i< fea.size(); i++){
        sum+= (fea[i] - mean)* (fea[i] - mean);
    }
    return sqrt(sum / fea.size());
}

double getGaussProb(double x,double mean, double sd){
    double temp = exp(-0.5 * ((x - mean)/sd)*((x - mean)/sd) );
    return (1/(2*PI*sd) ) * temp;
}

void isMatch(double f1, double f2, int cla){
    double clas_1 = classMean[0] * getGaussProb(f1, fea_1_mean[0], fea_1_sd[0])
            *getGaussProb(f2, fea_2_mean[0], fea_2_sd[0]);
    double clas_2 = classMean[1] * getGaussProb(f1, fea_1_mean[1], fea_1_sd[1])
            *getGaussProb(f2, fea_2_mean[1], fea_2_sd[1]);
    int found = -1;
//    cout<<clas_1<<" "<<clas_2<<endl;
    if (clas_1 > clas_2){
        found = 1;
    }else{
        found = 2;
    }
//    cout<<"ok"<<endl;
    if(cla == found){
        correct++;
//        cout<<"MATCH"<<endl;
    }
    else{
//        cout<<"NOT MATCH"<<endl;
        cout<<"Found : "<<found<<" Actually : "<<cla<<endl;
    }
}

int main(){
    ifstream file("train.txt");
    string line;
    double x, y, z;
    while(getline(file,line)){
        _count++;
        stringstream linestream(line);
        if(_count == 1){
            linestream >> classCount >> featureCount >> sampleCount;
            continue;
        }
        linestream >> x >> y >> z;
        if(z == 1){
//            classIndex_1.push_back(_count-2);
            class_1_count++;
            fea_1_1.push_back(x);
            fea_1_1_sum+= x;
            fea_2_1.push_back(y);
            fea_2_1_sum += y;
        }else{
            class_2_count++;
            fea_1_2.push_back(x);
            fea_1_2_sum+=x;
            fea_2_2.push_back(y);
            fea_2_2_sum +=y;
        }
    }
    fea_1_mean[0] = fea_1_1_sum / fea_1_1.size();
    fea_1_mean[1] = fea_1_2_sum / fea_1_2.size();
    fea_2_mean[0] = fea_2_1_sum / fea_2_1.size();
    fea_2_mean[1] = fea_2_2_sum / fea_2_2.size();

    fea_1_sd[0] = getSd(fea_1_1, fea_1_mean[0]);
    fea_1_sd[1] = getSd(fea_1_2, fea_1_mean[1]);
    fea_2_sd[0] = getSd(fea_2_1, fea_2_mean[0]);
    fea_2_sd[1] = getSd(fea_2_2, fea_2_mean[1]);

    classMean[0] = fea_1_1.size() / classCount;
    classMean[1] = fea_1_2.size() / classCount;
//    file("train.txt");
//    string line;
//    freopen("train.txt","r",stdin);
    ifstream file1("test.txt");
//    double x, y, z;
    cout<<"Testing.."<<endl;
    while(getline(file1,line)){
        stringstream linestream(line);
        totalTest++;
        linestream >> x >> y >> z;
        isMatch(x,y,z);
    }
    accuracy = (correct/totalTest)*100;
    cout<<"Accuracy: "<<accuracy<<"%"<<endl;
    return 0;
}
