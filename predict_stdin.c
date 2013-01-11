#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"

int print_null(const char *s,...) {}

static int (*info)(const char *fmt,...) = &printf;

struct feature_node *x;
int max_nr_attr = 64;

struct model* model_;
int flag_predict_probability=0;
int num_labels=0;
double threshold=0.1;

void exit_input_error(int line_num) {
       fprintf(stderr,"Wrong input format at line %d\n", line_num);
       exit(1);
}

void continue_input_error(int line_num) {
       fprintf(stderr,"Wrong input format at line %d\n", line_num);
}

static char *line = NULL;
static int max_line_len;

static char* readline() {
       int len;

       if(fgets(line,max_line_len,stdin) == NULL)
              return NULL;

       while(strrchr(line,'\n') == NULL) {
              max_line_len *= 2;
              line = (char *) realloc(line,max_line_len);
              len = (int) strlen(line);
              if(fgets(line+len,max_line_len-len,stdin) == NULL)
                     break;
       }
       return line;
}

void do_predict() {
       int correct = 0;
       int total = 0;
       double error = 0;
       double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

       int nr_class=get_nr_class(model_);
       double *prob_estimates=NULL;
       int j, n;
       int nr_feature=get_nr_feature(model_);
       if(model_->bias>=0)
              n=nr_feature+1;
       else
              n=nr_feature;

       if(flag_predict_probability) {
              int *labels;

              if(!check_probability_model(model_)) {
                     fprintf(stderr, "probability output is only supported for logistic regression\n");
                     exit(1);
              }

              labels=(int *) malloc(nr_class*sizeof(int));
              get_labels(model_,labels);
              prob_estimates = (double *) malloc(nr_class*sizeof(double));
//              fprintf(output,"labels");
//              for(j=0; j<nr_class; j++)
//                     fprintf(output," %d",labels[j]);
//              fprintf(output,"\n");
              free(labels);
       }

       max_line_len = 1024;
       line = (char *)malloc(max_line_len*sizeof(char));
       char *filter="-1";
       while(readline() != NULL) {
              int i = 0;
              double target_label, predict_label;
              char *idx, *val, *label, *endptr;
              char *mid;
              int inst_max_index = 0; // strtol gives 0 if wrong format

              mid = strtok(line,"|");

              label = strtok(NULL," \t\n");
              if(label == NULL) {// empty line
                     continue_input_error(total+1);
                     ++total;
                     continue;
              }

              target_label = strtod(label,&endptr);
              if(endptr == label || *endptr != '\0') {
                     continue_input_error(total+1);
                     ++total;
                     continue;
              }

              int success = 1;
              while(1) {
                     if(i>=max_nr_attr-2) { // need one more for index = -1
                            max_nr_attr *= 2;
                            x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
                     }

                     idx = strtok(NULL,":");
                     val = strtok(NULL," \t");

                     if(val == NULL)
                            break;
                     errno = 0;
                     x[i].index = (int) strtol(idx,&endptr,10);
                     if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index) {
                            continue_input_error(total+1);
                            success = 0;
                            break;
                     } else
                            inst_max_index = x[i].index;

                     errno = 0;
                     x[i].value = strtod(val,&endptr);
                     if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) {
                            continue_input_error(total+1);
                            success = 0;
                            break;
                     }
                     // feature indices larger than those in training are not used
                     if(x[i].index <= nr_feature)
                            ++i;
              }

              if(success ==0) {
                     ++total;
                     continue;
              }

              if(model_->bias>=0) {
                     x[i].index = n;
                     x[i].value = model_->bias;
                     i++;
              }
              x[i].index = -1;

              if(flag_predict_probability) {
                     int j;
                     predict_label = predict_probability(model_,x,prob_estimates);

                     //filter according to the max probability
                     int m=0;
                     double max=0.0;
                     for(m=0; m<model_->nr_class; m++) {
                            if(prob_estimates[m]>=max)
                                   max=prob_estimates[m];
                     }
                     //double threshold = 1.0/num_labels;
                     if(max<threshold) {
                            info("%s\t%g prob_estimate = %g (is too low )\n",mid,predict_label,max);
                            continue;
                     }

                     printf("%s\t%g %g",mid,predict_label,target_label);
                     for(j=0; j<model_->nr_class; j++)
                            printf(" %g",prob_estimates[j]);
                     printf("\n");
              } else {
                     predict_label = predict(model_,x);
                     printf("%g\n",predict_label);
              }

              if(label !="-1") {
                     if(predict_label == target_label)
                            ++correct;
                     error += (predict_label-target_label)*(predict_label-target_label);
                     sump += predict_label;
                     sumt += target_label;
                     sumpp += predict_label*predict_label;
                     sumtt += target_label*target_label;
                     sumpt += predict_label*target_label;
                     ++total;
              }
       }
       if(model_->param.solver_type==L2R_L2LOSS_SVR ||
          model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
          model_->param.solver_type==L2R_L2LOSS_SVR_DUAL) {
              info("Mean squared error = %g (regression)\n",error/total);
              info("Squared correlation coefficient = %g (regression)\n",
                   ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
                   ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
                  );
       } else {
              info("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
       }
       if(flag_predict_probability)
              free(prob_estimates);
}

void exit_with_help() {
       printf(
              "Usage: predict [options] test_file model_file output_file\n"
              "options:\n"
              "-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
              "-q : quiet mode (no outputs)\n"
       );
       exit(1);
}

int main(int argc, char **argv) {
       int i;

       // parse options
       for(i=1; i<argc; i++) {
              if(argv[i][0] != '-') break;
              ++i;
              switch(argv[i-1][1]) {
                     case 'b':
                            flag_predict_probability = atoi(argv[i]);
                            break;
                     case 't':
                            threshold= atof(argv[i]);
                            break;
                     case 'q':
                            info = &print_null;
                            i--;
                            break;
                     default:
                            fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
                            exit_with_help();
                            break;
              }
       }
       if(i>=argc)
              exit_with_help();

       if((model_=load_model(argv[i]))==0) {
              fprintf(stderr,"can't open model file %s\n",argv[i]);
              exit(1);
       }

       x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));

       do_predict();
       free_and_destroy_model(&model_);
       free(line);
       free(x);
       return 0;
}

