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
#include <algorithm>
#include <iomanip>
#include <float.h>
//#include <random>

using namespace std;

void usage(const char* prog);
vector<string> &split(const string &s, char delim, vector<std::string> &elems);
vector<string> split(const string &s, char delim);
void read_data_from_file(const char* file_name, vector<pair<int,vector<double> > >& data);
template <typename Iterator>
void prepare_paramters_for_calculations(
                        Iterator start, Iterator end,  int btsi, int tss,
                        map<int,vector<vector<double> > >& training_data_summary,
                        map<int,int>&,
                        map<int,vector<double> >&,
                        map<int,int>&,
                        unsigned int&);

void calculate_mean_variance_multinomialSums(
                                        int verbose,
                                        unsigned int& total_number_of_examples,
                                        map<int,vector<vector<double> > >& training_data_summary,
                                        map<int,int>& label_counts,
                                        map<int,double>& priors,
                                        map<int,vector<double> >& multinomial_likelihoods,
                                        map<int,int>& multinomial_sums,
                                        map<int,vector<double> >& sum_feature_values_per_label,
                                        map<int,vector<double> >& means,
                                        map<int,vector<double> >& variances,
                                        double& alpha);

template <typename Iterator>
double calculate_accuracy_of_test_set(
                                     int verbose,
                                     Iterator start, Iterator end,
                                     int& decision,
                                     map<int,double>& priors,
                                     map<int,vector<double> >& multinomial_likelihoods,
                                     map<int,vector<double> >& means,
                                     map<int,vector<double> >& variances);

void calculate_confidence_interval(vector<double> errors_in_k_fold_CV, double& mean, double& range);




int main(int argc, const char* argv[]){
    srand ( unsigned ( std::time(0) ) );

    // Smoothing factor
    double alpha = 1.0;
    // Decision rule
    int decision = 1;
    // Verbose
    int verbose = 0;

    // K in k-fold CV
    int k = 10;

    if(argc < 2){
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
            if(string(argv[i]) == "-k"){
                k = atoi(argv[i+1]);
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
    cout << "# K in K-fold CV:   " << k << endl;
    //cout << "# alpha param:   " << alpha << endl;
    //cout << "# training data: " << argv[argc-1] << endl << endl;
    cout << endl;


    //set precision for doubles in cout in the "whole" program
    //std::cout << std::fixed;
    //std::cout << std::setprecision(3);


    vector<pair<int,vector<double> > > all_data;
    read_data_from_file(argv[argc-1], all_data);
    random_shuffle (all_data.begin(), all_data.end());

    //k fold cross validation
    int num_examples = all_data.size();
    int tss = num_examples/k; // test set size, i.e. a window which will be slided to the right in k-fold cross validation to determin where the test is.
    vector<double> errors_in_k_fold_CV;
    for (int i = 0; i < k; i++) {
        if(verbose) cout << "------------ fold " << i+1  << " ------------------" << endl;
        int btsi = i*tss; //begin_test_set_index
        if(verbose) cout << "examples in test set: " << btsi << "-" << btsi+tss << endl;


        //begin training on this fold
        unsigned int total_number_of_examples = 0;
        map<int,vector<vector<double> > > training_data_summary;
        map<int,int> label_counts;
        map<int,double> priors;
        map<int,vector<double> > multinomial_likelihoods;
        map<int,int> multinomial_sums;
        map<int,vector<double> > sum_feature_values_per_label; //used to calculate mean for features per label
        map<int,vector<double> > means;
        map<int,vector<double> > variances;

        prepare_paramters_for_calculations(
                            all_data.begin(), all_data.end(), btsi, tss,
                            training_data_summary,
                            multinomial_sums,
                            sum_feature_values_per_label,
                            label_counts,
                            total_number_of_examples);

        //here I use iterator because I don't want to paye the cost of partitioning data into training and test sets
        calculate_mean_variance_multinomialSums(
                                                verbose,
                                                total_number_of_examples,
                                                training_data_summary,
                                                label_counts,
                                                priors,
                                                multinomial_likelihoods,
                                                multinomial_sums,
                                                sum_feature_values_per_label,
                                                means,
                                                variances,
                                                alpha);

        //calculate accuracy of this fold
        double accuracy = calculate_accuracy_of_test_set(
                                       verbose,
                                       all_data.begin()+btsi, all_data.begin()+(btsi+tss),
                                       decision,
                                       priors,
                                       multinomial_likelihoods,
                                       means,
                                       variances);


        errors_in_k_fold_CV.push_back(1-accuracy);

        /*
        //manual clean up
        training_data_summary.clear();
        label_counts.clear();
        priors.clear();
        multinomial_likelihoods.clear();
        multinomial_sums.clear();
        sum_feature_values_per_label.clear();
        means.clear();
        variances.clear();
        */



      }


      //calculate confidence interval
      double mean = 0.0;
      double range = 0.0;
      calculate_confidence_interval(errors_in_k_fold_CV, mean, range);
      cout << setprecision(2) << "\n\nCI of Errors: \n(" << 100*(mean-range) << "," << 100*(mean+range) << ")" << endl;
      cout << setprecision(2) << "center:" << mean*100 << "\nrange: " << range*100 << endl;




    return(0);

}


void usage(const char* prog){

   cout << "Read training data then perform cross validation using naive Bayes classifier:\nUsage:\n" << prog << " [options] training_set" << endl << endl;
   cout << "Options:" << endl;
   cout << "-d <int> Decsion rule. 1 = gaussian (default)" << endl;
   cout << "                       2 = multinomial" << endl;
   cout << "                       3 = bernoulli" << endl;
   cout << "-k       K in K-fold Cross Validation, default 10." << endl;
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
void read_data_from_file(const char* file_name, vector<pair<int,vector<double> > >& data) {

    ifstream fin(file_name);
    string line;
    bool first_row_read =false;
    while (getline(fin, line)){ //read a line from input file, i.e. row strating with label, and following with features values separataed with space
        if(line.length()){
            if(line[0] != '#' && line[0] != ' '){
                first_row_read = true;
                vector<string> tokens = split(line,' ');
                int label = int (tokens[0][0]);

                vector<double> values;
                for(unsigned int i = 1; i < tokens.size(); i++){  //extract feature i's value for this row
                    double val = atof(tokens[i].c_str());
                    //values.push_back(val);  //raw values
                    values.push_back( (val==0.0)? 0.1 : val*val );  //consider square values
                    //values.push_back( abs(val) );  //consider absolute values
                }

                pair<int,vector<double> > row;
                row.first = label;
                row.second = values;
                data.push_back(row);

                //check if number of features is fixed
                if( first_row_read ) {
                    if (values.size() != data[0].second.size()){
                      cout << "# inconsistent feature count! sparse data not supported yet." << endl;
                      cout << "# " << values.size() << " vs " << data[0].second.size() << endl;
                      cout << line << endl;
                      fin.close();
                      exit(1);
                    }
                }
            }
          }
    }
    fin.close();

}


//some pre-calculations so that we can easily calculate parameters in naive bayes
//Iterators: start and end are the begining and end of all data; btsi and etsi are the begin and end index for test set,thus we iterate from begining till btsi, and then from etsi+1 till end.
template <typename Iterator>
void prepare_paramters_for_calculations(
                         Iterator start, Iterator end,  int btsi, int tss,
                         map<int,vector<vector<double> > >& training_data,
                         map<int,int>& multinomial_sums,
                         map<int,vector<double> >& sum_feature_values_per_label,
                         map<int,int>& label_counts,
                         unsigned int& total_number_of_examples) {

      int counter = 0;
      for (Iterator row = start; row !=end; ++row) {
          //ignore test set
          if(counter == btsi) {
            row += tss-1 ;  //this should be "tss+1", but I have to handle boundaries (fix later)
          }
          counter ++;

          int label = (*row).first;
          for(unsigned int i = 0; i < (*row).second.size(); i++){  //extract feature i's value for this row
              if(sum_feature_values_per_label.find(label) == sum_feature_values_per_label.end()){ // if this label has not been seen
                  vector<double> empty;
                  for(unsigned int j = 0; j < (*row).second.size(); j++){
                      empty.push_back(0.0);
                  }
                  sum_feature_values_per_label[label] = empty;
              }
              sum_feature_values_per_label[label][i] += (*row).second[i];
              multinomial_sums[label] += (*row).second[i];
          }

          training_data[label].push_back((*row).second);
          label_counts[label]++;
          total_number_of_examples++;
      }
}

//parameter "data" passed here is actually "training_data"
void calculate_mean_variance_multinomialSums(
                                        int verbose,
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

        if(verbose) {
          if(it->first > 0){
              cout << "class " << char(it->first) << ", prior: " << priors[it->first] << endl;
          }else{
            cout << "class " << char(it->first) << ", prior: " << priors[it->first] << endl;
              //printf ("class %i prior: %1.3f\n",it->first,priors[it->first]);
          }
          cout << "feature\tmean\tvar\tstddev" << endl;
        }


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

        if(verbose) {
          for(unsigned int i = 0; i < feature_means.size(); i++){
              printf("%i\t%2.3f\t%2.3f\t%2.3f\n",i+1,feature_means[i],feature_variances[i],sqrt(feature_variances[i]));
              //cout << feature_means[i] << "\t" << sqrt(feature_variances[i]) << endl;
          }
        }

        means[it->first] = feature_means;
        variances[it->first] = feature_variances;

    }

}


//returns accuracy as number in [0,1]
//I pass range iterators because I don't want to pay the cost of partitioning data vector to trainign and test data.
template <typename Iterator>
double calculate_accuracy_of_test_set(
                                     int verbose,
                                     Iterator start, Iterator end,
                                     int& decision,
                                     map<int,double>& priors,
                                     map<int,vector<double> >& multinomial_likelihoods,
                                     map<int,vector<double> >& means,
                                     map<int,vector<double> >& variances){

       // Classify
       if(verbose) cout << "\nClassifying:" << endl;
       if(verbose) cout << "class\tlog_posterior_numer" << endl;
       int correct = 0;
       int total = 0;

       //for (const auto& row : test_data){
       for (Iterator row = start; row !=end; ++row){
             int label = (*row).first;
             //row.second is the vector containing valuse of this row (label)



             /*
             //for NB-2
             double second_best_likelihood = 0.0;
             double third_best_likelihood = 0.0;
             int second_best_label = -1;
             int third_best_label = -1;
             */

             int predlabel = 0;
             double maxlikelihood = -DBL_MAX;
             //double denom = 0.0;
             vector<double> probs;
             for(auto it = priors.begin(); it != priors.end(); it++){
                 //double numer = priors[it->first];
                 double numer = log(priors[it->first]);
                 for(unsigned int j = 0; j < (*row).second.size(); j++){
                     switch(decision){
                         case 2:
                             // Multinomial
                             if((*row).second[j]){
                                 numer *= pow(multinomial_likelihoods[it->first][j],(*row).second[j]);
                             }
                             break;
                         case 3:
                             // Bernoulli
                             numer *= (pow(means[it->first][j],(*row).second[j]) * pow((1.0-means[it->first][j]),(1.0-(*row).second[j])));
                             break;
                         default:
                             // Gaussian
                             //numer *= (1/sqrt(2*M_PI*variances[it->first][j]))*exp((-1*((*row).second[j]-means[it->first][j])*((*row).second[j]-means[it->first][j]))/(2*variances[it->first][j]));
                             //numer += log( (1/sqrt(2*M_PI*variances[it->first][j]))*exp((-1*((*row).second[j]-means[it->first][j])*((*row).second[j]-means[it->first][j]))/(2*variances[it->first][j])) );  //normal distribution
                             numer += log( (1/sqrt(2*M_PI*(*row).second[j]))*exp(-1*((*row).second[j]/2)) );  //for squares: chi-square distribution
                             //numer += log( (1/sqrt(2*M_PI*variances[it->first][j]))*exp((-1*((*row).second[j]-means[it->first][j])*((*row).second[j]-means[it->first][j]))/(2*variances[it->first][j])) + (1/sqrt(2*M_PI*variances[it->first][j]))*exp((-1*((*row).second[j]+means[it->first][j])*((*row).second[j]-means[it->first][j]))/(2*variances[it->first][j])));  //folded distribution
                             break;
                     }
                 }

                 if(verbose){
                     char current_label = char(it->first);
                     cout << current_label << ":\t" << numer << endl;
                 }

                 if(numer > maxlikelihood){
                     maxlikelihood = numer;
                     predlabel = it->first;
                 }
                 /*
                //for NB-2
                 else {
                     //keep track of 2nd and 3rd best
                     if(numer > second_best_likelihood) {
                       second_best_likelihood = numer;
                       second_best_label = it->first;
                     }else {
                         if(numer > third_best_likelihood) {
                           third_best_likelihood = numer;
                           third_best_label = it->first;
                         }
                     }
                 }
                 */

                 //denom += numer;



                 probs.push_back(numer);
             }
             //for(unsigned int j = 0; j < probs.size(); j++){
             //    cout << probs[j]/denom << " ";
             //}


             if(verbose){
               char pred_l = char(predlabel);
               char lbl = char(label);
               cout << "label: " << lbl << "\tpredictaed_label:" << pred_l << "\t";
             }


            /*
            //random classifier
            //labels with 0 frequencies are excluded
            std::vector<char>  all_labels {'J', 'A', 'K', 'L', 'D', 'V', 'T', 'M', 'N', 'U', 'O', 'C', 'G', 'E', 'F', 'H', 'I', 'P', 'Q'};
            int l = all_labels.size();
            int random_label = all_labels[rand()%l];
            predlabel = int(random_label);
            */

             if(predlabel == label){
                 if(verbose) cout << "\tcorrect" << endl << endl;
                 correct++;
             /*
             //pick top three choices in NB
             }else if(second_best_label == label) {
                 if(verbose) cout << "\tcorrect - 2" << endl << endl;
                 correct++;
             }else if (third_best_label == label){
                 if(verbose) cout << "\tcorrect - 3" << endl << endl;
                 correct++;
             */
             }
             else{
                 if(verbose) cout << "\tincorrect" << endl << endl;
             }

             total++;
       }


       double accuracy = ((double)correct/total);
       printf ("Accuracy: %3.2f %% (%i/%i)\n", (100*accuracy),correct,total);
       return accuracy;

}


//paramert "range" will be (2.262 * SE) where 2.262 comes from 95% CI and k = 10
void calculate_confidence_interval(vector<double> errors_in_k_fold_CV, double& mean, double& range) {

    int k = errors_in_k_fold_CV.size();

    //calculate average error
    double average_error = 0;
    for(auto i : errors_in_k_fold_CV) {
      average_error += i;
    }
    average_error /= k;
    mean = average_error;

    //calulate variance of errors
    double variance = 0;
    for(auto i : errors_in_k_fold_CV) {
      variance += (i - average_error)*(i - average_error);
    }
    variance /= (k-1);

    //calculate standard error
    double standard_deviation_of_errors = sqrt(variance);  // describes the spread of values in the sample (i.e. errors here)
    double standard_error =  standard_deviation_of_errors / sqrt(k);   //This is the standard deviation of the sample mean, xBar, and describes its accuracy as an estimate of the population mean, mu.

    //when k=10 and we want 95% CI, 2.262
    range = 2.262 * standard_error;

}
