import csv
import copy
import math
from os import WIFCONTINUED, error
import re
import numpy as np

'''
input: Training Examples; Iteration number T; Attributes
Output: Final hypothesis: [[vote, split_attr, [value lst that the output is yes]]]
'''
class Boosting:
    examples = ''
    test_examples = ''
    attrs = ''
    iterations = 0
    final_hypothesis = []
    training_error = []         #using boosting
    test_error = []
    training_current_error = []         #only using the current classfier
    test_current_error = []

    def __init__(self, examples, test_examples, attrs, iterations):
        # print('enter Boosting class')
        self.examples = examples
        self.attrs = attrs
        self.test_examples = test_examples
        self.iterations = iterations
    
    # get examples that satisfy: attr = attr_v
    def get_sub_examples(self, examples, attr, attr_v):
        sub_examples = []
        for example in examples:
            if example[attr] == attr_v:
                sub_examples.append(example)
        return sub_examples

    # test if the sub_example has the same label
    def has_same_label(self, examples):
        label = examples[0]['label']
        for example in examples:
            if example['label'] != label:
                return False
        return True
    
    # get the most common label
    def get_most_common_label(self, examples):
        # get the most common label
        label_num = [0, 0]
        for example in examples:
            lb = example['label']
            if lb == 'yes':
                label_num[0] += example['weight']
            elif lb == 'no':
                label_num[1] += example['weight']

        if label_num[0] > label_num[1]:
            return 'yes'
        else:
            return 'no'
    
    # calculate the information gain
    def get_gain(self, examples, attr, values):
        # get the proportions of different label values
        def get_label_prop_dict(sub_examples):
            res = {}
            total = 0
            for example in sub_examples:
                total += example['weight']
                if example['label'] in res.keys():
                    res[example['label']] += example['weight']
                else:
                    res[example['label']] = example['weight']
            for key in res:
                res[key] /= total
            return res

        def Entropy(sub_examples):
            res = 0
            label_prop_dict = get_label_prop_dict(sub_examples)
            for key in label_prop_dict:
                val = label_prop_dict[key]
                res -= val * math.log(val, 2)
            return res

        # purity: Examples
        purity_examples = Entropy(examples)

        # purity: subExamples
        purity_values = []
        proportions = []
        examples_all_weight = 0
        for example in examples:
            examples_all_weight += example['weight']

        for value in values:
            sub_examples = []
            sub_examples_all_weight = 0
            for example in examples:
                if example[attr] == value:
                    sub_examples_all_weight += example['weight']
                    sub_examples.append(example)               
            purity_values.append(Entropy(sub_examples))

            proportions.append(sub_examples_all_weight/examples_all_weight)
        expected_purity = 0

        for purity_value, proportion in zip(purity_values, proportions):
            expected_purity += purity_value*proportion

        gain = purity_examples - expected_purity
        return gain

    # find the best attributes to split the examples
    def get_best_attr(self, examples, attrs):
        gains = {}
        # calculate information gains for all attrs
        for attr, values in attrs.items():
            gains[attr] = self.get_gain(examples, attr, values)
        # select the best one
        best_attr = ''
        best_attr_gain = -10
        for attr, gain in gains.items():
            if gain > best_attr_gain:
                best_attr_gain = gain
                best_attr = attr
        return best_attr

    # get error for this iteration
    def get_error(self, examples, selected_attr, true_attr_val_lst):
        error = 0.5
        y = ''
        h = ''
        weight = ''

        for example in examples:
            weight = example['weight']
            if example['label'] == 'yes':
                y = 1
            elif example['label'] == 'no':
                y = -1
            if example[selected_attr] in true_attr_val_lst:
                h = 1
            else:
                h = -1
            
            error -= 0.5*(weight*h*y)
        
        return error
    
    # get the updated weights
    def get_update_weights(self, examples, selected_attr, true_attr_val_lst, vote):
        y = ''
        h = ''
        weight = ''
        updated_weights = []

        for example in examples:
            weight = example['weight']
            if example['label'] == 'yes':
                y = 1
            elif example['label'] == 'no':
                y = -1
            if example[selected_attr] in true_attr_val_lst:
                h = 1
            else:
                h = -1
            updated_weights.append(weight*(math.exp(-1*vote*y*h)))
        
        # normalization
        Z = 0
        for updated_weight in updated_weights:
            Z += updated_weight
        for i in range(len(updated_weights)):
            updated_weights[i] /= Z
        
        return updated_weights

    # get [value lst that the output is yes]
    def get_current_hypothsis(self, examples, attr):
        lst = []
        # for each value of the attr, get a sub_example, and generate a branch for it
        for attr_v in self.attrs[attr]:
            label = '' 
            sub_examples = self.get_sub_examples(examples, attr, attr_v)
            # if the subset is empty, then make the subset as a leaf, and set the most common label
            if len(sub_examples) == 0:
                label = self.get_most_common_label(examples)            
            # else if the subset has the same label, make the subset as a leaf and set the common label
            elif self.has_same_label(sub_examples):
                label = sub_examples[0]['label']
            else:
                label = self.get_most_common_label(sub_examples)
            
            if label == 'yes':
                lst.append(attr_v)
        return lst

    # test dataset, hypothsis_option:1-use current boosting; 2-using the last tree trump
    def testing(self, test_data, hypothsis_option):
        def get_hypothesis_from_boosting(example):
            values = 0
            for hypothisis in self.final_hypothesis:
                selected_attr = hypothisis[1]
                vote = hypothisis[0]
                true_attr_value_lst = hypothisis[2]
                h = ''
                if example[selected_attr] in true_attr_value_lst:
                    h = 1
                else:
                    h = -1
                values += vote*h
            return values
        
        def get_hypothesis_from_current(example):
            hypothisis = self.final_hypothesis[-1]
            selected_attr = hypothisis[1]
            true_attr_value_lst = hypothisis[2]
            h = ''
            if example[selected_attr] in true_attr_value_lst:
                h = 1
            else:
                h = -1
            return h

        num = len(test_data)
        correct_num = 0
        for example in test_data:
            y = example['label']
            h = ''
            if hypothsis_option == 1:
                h = get_hypothesis_from_boosting(example)
            elif hypothsis_option == 2:
                h = get_hypothesis_from_current(example)
            if (h > 0 and y == 'yes') or (h < 0 and y == 'no'):
                correct_num += 1
        return 1-correct_num/num

    # return the final hypothsis
    def get_final_hypothesis(self):
        # print('enter the boosting algorithm')
        # number of examples
        num = len(self.examples)
        # weights for each example
        weights = []
        for i in range(num):
            weights.append(1/num)
        for i in range(num):
            self.examples[i]['weight'] = weights[i]

        for round in range(0, self.iterations):
            print('round:', round)
            # find a hypothsis h - select a best feature to split the group
            selected_attr = self.get_best_attr(self.examples, self.attrs)           # the selected attr
            # print('selected_attr:', selected_attr)
            # get the attr_values of attr that is ture
            true_attr_val_lst = self.get_current_hypothsis(self.examples, selected_attr)
            # print('true_attr_val_lst:', true_attr_val_lst)

            # compute the error
            error = self.get_error(self.examples, selected_attr, true_attr_val_lst)
            # print('error:', error)

            # compute the vote
            vote = 0.5*math.log((1-error)/error)
            # print('vote:', vote)

            # update the weights
            updated_weights = self.get_update_weights(self.examples, selected_attr, true_attr_val_lst, vote)
            for i in range(num):
                self.examples[i]['weight'] = updated_weights[i]
            
            # add the current hypothsis into this final hypothsis
            self.final_hypothesis.append([vote, selected_attr, true_attr_val_lst])

            # test part
            self.training_error.append(self.testing(self.examples, 1))
            self.test_error.append(self.testing(self.test_examples, 1))
            self.training_current_error.append(self.testing(self.examples, 2))
            self.test_current_error.append(self.testing(self.test_examples, 2))

        return self.final_hypothesis

    # store the testing result in to csv
    def store_test_result(self):
        # wirte the training and test error for the boosting
        res1 = [['training error', 'test error']]
        for i in range(len(self.training_error)):
            res1.append([self.training_error[i], self.test_error[i]])
        with open('./Results/result1.csv', 'w') as f:
            f_csv = csv.writer(f, lineterminator='\n')
            f_csv.writerows(res1)
        print('store in result1.csv!')
        
        # wirte the training and test error for the current classifier
        res2 = [['training_ error', 'test_ error']]
        for i in range(len(self.training_current_error)):
            res2.append([self.training_current_error[i], self.test_current_error[i]])
        with open('./Results/result2.csv', 'w') as f:
            f_csv = csv.writer(f, lineterminator='\n')
            f_csv.writerows(res2)
        print('store in result2.csv!')


'''
generate the training example
example = [{'attr1': , 'attr2': , 'attr3': }, ...],
'''
def process_data(file_name):
    examples = []
    with open(file_name, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            example = {}
            example['age'] = row[0]
            example['job'] = row[1]
            example['marital'] = row[2]
            example['education'] = row[3]
            example['default'] = row[4]
            example['balance'] = row[5]
            example['housing'] = row[6]
            example['loan'] = row[7]
            example['contact'] = row[8]
            example['day'] = row[9]
            example['month'] = row[10]
            example['duration'] = row[11]
            example['campaign'] = row[12]
            example['pdays'] = row[13]
            example['previous'] = row[14]
            example['poutcome'] = row[15]
            example['label'] = row[16]
            example['weight'] = 0    # default 
            examples.append(example)
    return examples


# find the media of each numerical attribute
def find_medias(examples):
    numerical_media_map = {'age': '', 'balance': '', 'day': '', 'duration': '', 'campaign': '', 'pdays': '', 'previous': ''}
    for key in numerical_media_map.keys():
        numerical_lst = []
        for example in examples:
            numerical_lst.append(int(example[key]))
        numerical_media_map[key] = np.median(numerical_lst)
    return numerical_media_map


# map numrical attribute to binary 
def num_to_binary(media_map, examples):
    for example in examples:
        for attr, media in media_map.items():
            value = int(example[attr])
            if value > media or value == media:
                example[attr] = 'yes'
            else:
                example[attr] = 'no'
    return examples


if __name__ == '__main__':
    # process training dataset, transfrom numerial attributes
    train_examples = process_data('./data/bank/train.csv')
    numerical_media_map = find_medias(train_examples)
    train_examples = num_to_binary(numerical_media_map, train_examples)
    
    # process test dataset, transfrom numerial attributes
    test_examples = process_data('./data/bank/test.csv')
    test_examples = num_to_binary(numerical_media_map, test_examples)

    # attributes; big than or equal to the media =>yes, else no
    attrs = {
        'age': ["yes","no"],
        'job': ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
        "blue-collar", "self-employed", "retired", "technician", "services"],
        'marital': ["married", "divorced", "single"],
        'education': ["unknown","secondary","primary","tertiary"],
        'default': ["yes","no"],
        'balance': ["yes","no"],
        'housing': ["yes","no"],
        'loan': ["yes","no"],
        'contact': ["unknown","telephone","cellular"],
        'day': ["yes","no"],
        'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        'duration': ["yes","no"],
        'campaign': ["yes","no"],
        'pdays': ["yes","no"],
        'previous': ["yes","no"],
        'poutcome': ["unknown","other","failure","success"]
    }

    # boosting algorithm setting
    T = 500   # the number of iterations

    # get the final hypothesis
    boosting = Boosting(train_examples, test_examples, attrs, T)
    final_hypothesis = boosting.get_final_hypothesis()

    # do the test
    print('final result')
    print(final_hypothesis)
    print(boosting.training_error)
    print(boosting.training_current_error)
    print(boosting.test_error)
    print(boosting.test_current_error)

    # generate the result files
    boosting.store_test_result()


''''
Boosting ????????????????????????

1.??????????????????????????????????????????????????????item???????????????dict?????????????????????????????????????????????????????????weight??????????????????????????????
2.???????????????????????????????????????????????????boosting??????
    1. ?????????????????????????????????????????????????????????
    2. ?????????????????????
        1. ??????information gain???????????????feature?????????
        2. ??????????????????????????????????????????????????????????????????????????????????????????????????????yes??????????????????????????????no???
        3. ??????????????????????????????????????????????????????????????????
        4. ??????????????????????????????????????????vote
        5. ????????????????????????????????????item????????????
        6. ????????????item??????????????????????????????
    3. ?????????????????????, ?????????1??? ??????2??? ????????????????????????1 =???vote??? ????????????????????????label???yes???????????????
3.????????????
    1. ??????????????????????????????????????????
    2. ?????????????????????????????????????????????????????????

?????????
???1????????????????????????????????????????????????????????????????????????1-500

???2????????????????????????500???????????????????????????????????????
'''