// Naive Bayes Implementation
// (c) Tim Nugent 2014
// timnugent@gmail.com

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <map>

using namespace std;

void usage(const char* prog);
vector<string> &split(const string &s, char delim, vector<std::string> &elems);
vector<string> split(const string &s, char delim);
void read_data_from_file(
                        const char*,
                        map<int,vector<vector<double> > >&,
                        map<int,int>&,
                        map<int,vector<double> >&,
                        map<int,int>&,
                        unsigned int&);
void calculate_mean_variance_multinomialSums(
                                        unsigned int& total_number_of_examples,
                                        map<int,vector<vector<double> > >& data,
                                        map<int,int>& label_counts,
                                        map<int,double>& priors,
                                        map<int,vector<double> >& multinomial_likelihoods,
                                        map<int,int>& multinomial_sums,
                                        map<int,vector<double> >& sum_feature_values_per_label,
                                        map<int,vector<double> >& means,
                                        map<int,vector<double> >& variances,
                                        double& alpha);





int main(int argc, const char* argv[]){

    // Smoothing factor
    double alpha = 1.0;
    // Decision rule
    int decision = 1;
    // Verbose
    int verbose = 0;

    if(argc < 3){
        usage(argv[0]);
        return(1);
    }else{
        cout << "# called with: ";
        for(int i = 0; i < argc; i++){
            cout << argv[i] << " ";
            if(string(argv[i]) == "-d" && i < argc-1){
                decision = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-a" && i < argc-1){
                alpha = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-v"){
                verbose = 1;
            }
            if(string(argv[i]) == "-h"){
                usage(argv[0]);
                return(1);
            }
        }
        cout << endl;
    }
    switch(decision){
        case 2:
            cout << "# decision rule: multinomial" << endl;
            break;
        case 3:
            cout << "# decision rule: bernoulli" << endl;
            break;
        default:
            cout << "# decision rule: gaussian" << endl;
            break;
    }
    cout << "# alpha param:   " << alpha << endl;
    cout << "# training data: " << argv[argc-2] << endl;
    cout << "# test data:     " << argv[argc-1] << endl;

    unsigned int total_number_of_examples = 0;
    map<int,vector<vector<double> > > data;
    map<int,int> label_counts;
    map<int,double> priors;
    map<int,vector<double> > multinomial_likelihoods;
    map<int,int> multinomial_sums;
    map<int,vector<double> > sum_feature_values_per_label; //used to calculate mean for features per label
    map<int,vector<double> > means;
    map<int,vector<double> > variances;


    read_data_from_file(
                        argv[argc-2],
                        data,
                        multinomial_sums,
                        sum_feature_values_per_label,
                        label_counts,
                        total_number_of_examples);

    calculate_mean_variance_multinomialSums(
                                            total_number_of_examples,
                                            data,
                                            label_counts,
                                            priors,
                                            multinomial_likelihoods,
                                            multinomial_sums,
                                            sum_feature_values_per_label,
                                            means,
                                            variances,
                                            alpha);





    // Classify
    cout << "Classifying:" << endl;
    if(verbose) cout << "class\tprob\tresult" << endl;
    int correct = 0;
    int total = 0;

    string line;
    ifstream fin;
    fin.open(argv[argc-1]);
    while (getline(fin, line)){
        if(line[0] != '#' && line[0] != ' ' && line[0] != '\n'){
            vector<string> tokens = split(line,' ');
            vector<double> values;
            int label = int (tokens[0][0]);

            for(unsigned int i = 1; i < tokens.size(); i++){
                values.push_back(atof(tokens[i].c_str()));
            }

            int predlabel = 0;
            double maxlikelihood = 0.0;
            double denom = 0.0;
            vector<double> probs;
            for(auto it = priors.begin(); it != priors.end(); it++){
                double numer = priors[it->first];
                for(unsigned int j = 0; j < values.size(); j++){
                    switch(decision){
                        case 2:
                            // Multinomial
                            if(values[j]){
                                numer *= pow(multinomial_likelihoods[it->first][j],values[j]);
                            }
                            break;
                        case 3:
                            // Bernoulli
                            numer *= (pow(means[it->first][j],values[j]) * pow((1.0-means[it->first][j]),(1.0-values[j])));
                            break;
                        default:
                            // Gaussian
                            numer *= (1/sqrt(2*M_PI*variances[it->first][j])*exp((-1*(values[j]-means[it->first][j])*(values[j]-means[it->first][j]))/(2*variances[it->first][j])));
                            break;
                    }
                }
                /*
                if(verbose){
                    if(it->first > 0){
                        cout << "+" << it->first << ":" << numer << endl;
                    }else{
                        cout << it->first << ":" << numer << endl;
                    }
                }
                */
                if(numer > maxlikelihood){
                    maxlikelihood = numer;
                    predlabel = it->first;
                }
                denom += numer;
                probs.push_back(numer);
            }
            //for(unsigned int j = 0; j < probs.size(); j++){
            //    cout << probs[j]/denom << " ";
            //}

            if(verbose){
                if(predlabel > 0){
                    printf ("+%i\t%1.3f\t", predlabel,(maxlikelihood/denom));
                }else{
                    printf ("%i\t%1.3f\t", predlabel,(maxlikelihood/denom));
                }
            }
            if(label){
                if(predlabel == label){
                    if(verbose) cout << "correct" << endl;
                    correct++;
                }else{
                    if(verbose) cout << "incorrect" << endl;
                }
            }else{
                if(verbose) cout << "<no label>" << endl;
            }
            total++;
        }
    }
    fin.close();
    printf ("Accuracy: %3.2f %% (%i/%i)\n", (100*(double)correct/total),correct,total);

    return(0);

}



void usage(const char* prog){

   cout << "Read training data then classify test data using naive Bayes:\nUsage:\n" << prog << " [options] training_data test_data" << endl << endl;
   cout << "Options:" << endl;
   cout << "-d <int> Decsion rule. 1 = gaussian (default)" << endl;
   cout << "                       2 = multinomial" << endl;
   cout << "                       3 = bernoulli" << endl;
   cout << "-a       Smoothing parameter alpha. default 1.0 (Laplace)" << endl;
   cout << "-v       Verbose." << endl << endl;
}


vector<string> &split(const string &s, char delim, vector<std::string> &elems) {
    stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}


//assuming file_name contains space delimited rows, and the first column is a single CHAR represnting the class label
void read_data_from_file(
                         const char* file_name,
                         map<int,vector<vector<double> > >& data,
                         map<int,int>& multinomial_sums,
                         map<int,vector<double> >& sum_feature_values_per_label,
                         map<int,int>& label_counts,
                         unsigned int& total_number_of_examples) {

ifstream fin(file_name);
//fin.ignore();
string line;
while (getline(fin, line)){ //read a line from input file, i.e. row strating with label, and following with features values separataed with space
    if(line.length()){
        if(line[0] != '#' && line[0] != ' '){
            vector<string> tokens = split(line,' ');
            vector<double> values;
            //int label = atoi(tokens[0].c_str());  //first column is label
            int label = int (tokens[0][0]);  //first column is label
            //cout << "label : " << tokens[0] << " -> " << label << endl;

            for(unsigned int i = 1; i < tokens.size(); i++){  //extract feature i's value for this row
                values.push_back(atof(tokens[i].c_str()));
                if(sum_feature_values_per_label.find(label) == sum_feature_values_per_label.end()){ // if this label has not been seen
                    vector<double> empty;
                    for(unsigned int j = 1; j < tokens.size(); j++){
                        empty.push_back(0.0);
                    }
                    sum_feature_values_per_label[label] = empty;
                }
                sum_feature_values_per_label[label][i-1] += values[i-1];
                multinomial_sums[label] += values[i-1];
            }
            //check if number of features is fixed
            if(values.size() != sum_feature_values_per_label[label].size()){
                cout << "# inconsistent feature count! sparse data not supported yet." << endl;
                cout << "# " << values.size() << " vs " << sum_feature_values_per_label[label].size() << endl;
                cout << line << endl;
                fin.close();
                exit(1);
            }

            data[label].push_back(values);
            label_counts[label]++;
            total_number_of_examples++;
        }
    }
}
fin.close();

}



void calculate_mean_variance_multinomialSums(
                                        unsigned int& total_number_of_examples,
                                        map<int,vector<vector<double> > >& data,
                                        map<int,int>& label_counts,
                                        map<int,double>& priors,
                                        map<int,vector<double> >& multinomial_likelihoods,
                                        map<int,int>& multinomial_sums,
                                        map<int,vector<double> >& sum_feature_values_per_label,
                                        map<int,vector<double> >& means,
                                        map<int,vector<double> >& variances,
                                        double& alpha) {

    for(auto it = sum_feature_values_per_label.begin(); it != sum_feature_values_per_label.end(); it++){

        priors[it->first] = (double)label_counts[it->first]/total_number_of_examples;


        if(it->first > 0){
            cout << "class " << char(it->first) << ", prior: " << priors[it->first] << endl;
        }else{
          cout << "class " << char(it->first) << ", prior: " << priors[it->first] << endl;
            //printf ("class %i prior: %1.3f\n",it->first,priors[it->first]);
        }
        cout << "feature\tmean\tvar\tstddev\tmnl" << endl;


        // Calculate means
        vector<double> feature_means;
        for(unsigned int i = 0; i < it->second.size(); i++){
            feature_means.push_back(sum_feature_values_per_label[it->first][i]/label_counts[it->first]);
            //cout << "mean for label " << char(it->first) << ", and feature " << i <<  " is : " << feature_means[i] << endl;
        }

        // Calculate variances
        vector<double> feature_variances(feature_means.size());
        for(unsigned int i = 0; i < data[it->first].size(); i++){
            for(unsigned int j = 0; j < data[it->first][i].size(); j++){
                feature_variances[j] += (data[it->first][i][j]-feature_means[j])*(data[it->first][i][j]-feature_means[j]);
            }
        }
        for(unsigned int i = 0; i < feature_variances.size(); i++){
            feature_variances[i] /= data[it->first].size();
            //cout << "variance for label " << char(it->first) << ", and feature " << i <<  " is : " << feature_variances[i] << endl;
        }

        // Calculate multinomial likelihoods
        for(unsigned int i = 0; i < feature_means.size(); i++){
            double mnl = (sum_feature_values_per_label[it->first][i]+alpha)/(multinomial_sums[it->first]+(alpha*feature_means.size()));
            //cout << sum_feature_values_per_label[it->first][i] << " + 1 / " << multinomial_sums[it->first] << " + " << feature_means.size() << endl;
            multinomial_likelihoods[it->first].push_back(mnl);
        }


        for(unsigned int i = 0; i < feature_means.size(); i++){
            printf("%i\t%2.3f\t%2.3f\t%2.3f\t%2.3f\n",i+1,feature_means[i],feature_variances[i],sqrt(feature_variances[i]),multinomial_likelihoods[it->first][i]);
            //cout << feature_means[i] << "\t" << sqrt(feature_variances[i]) << endl;
        }

        means[it->first] = feature_means;
        variances[it->first] = feature_variances;

    }

}
