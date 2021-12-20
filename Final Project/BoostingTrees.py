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

# test dataset, hypothsis_option:1-use current boosting; 2-using the last tree trump
    def get_test_result(self, test_data, order):
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

        num = 1
        result = [['ID', 'Prediction']]

        for example in test_data:
            h = get_hypothesis_from_boosting(example)
            if h > 0:
                result.append([num, 1])
            elif h < 0:
                result.append([num, 0])
            num += 1
        
        with open('./Results/resultBoosting'+str(order)+'.csv', 'w') as f:
            csv_f = csv.writer(f, lineterminator='\n')
            csv_f.writerows(result)

        

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
            e = self.testing(self.examples, 1)
            self.training_error.append(e)
            self.get_test_result(self.test_examples, round)
            print(e)
            # write into dataset

            # self.test_error.append(self.testing(self.test_examples, 1))
            # self.training_current_error.append(self.testing(self.examples, 2))
            # self.test_current_error.append(self.testing(self.test_examples, 2))

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

# get the training data
def get_missing_data(examples):
    return {'age': '38.7', 'workclass': 'Private', 'fnlwgt': '189000', 'education': 'HS-grad', 'education-num': '10', 
    'marital-status': 'Married-civ-spouse', 'occupation': 'Prof-specialty', 'relationship': 'Husband', 'race': 'White', 
    'sex': 'Male', 'capital-gain': '0', 'capital-loss': '87.2', 'hours-per-week': '40', 'native-country': 'United-States', 'label': '1'}

# update the missing data
def update_missing_data(missing_data, examples):
    num = 0
    for example in examples:
        for key in example.keys():
            if example[key] == '?':
                num += 1
                example[key] = missing_data[key]
    print(num)
    return examples

def get_continuous_to_category_data():
    return {'age': [28, 37, 48], 
    'fnlwgt': [86190, 160000, 234000], 
    'education-num': [9, 10, 12], 
    'capital-gain': [1], 
    'capital-loss': [1], 
    'hours-per-week': [35, 42]}

# make continuous attr become category one
def continuous_to_category(continuous_to_category, examples):
    for example in examples:
        for key in continuous_to_category.keys():
            val = int(example[key])
            mapping = continuous_to_category[key]
            category = 0
            
            for num in mapping:
                if val <= num:
                    example[key] = str(category)+'a'
                    break
                category += 1
            if category == len(mapping):
                example[key] = str(category)+'a'

    return examples


'''
generate the training example
example = [{'attr1': , 'attr2': , 'attr3': }, ...],
'''
def process_training_data(file_name):
    examples = []
    with open(file_name, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            example = {}
            example['age'] = row[0]
            example['workclass'] = row[1]
            example['fnlwgt'] = row[2]
            example['education'] = row[3]
            example['education-num'] = row[4]
            example['marital-status'] = row[5]
            example['occupation'] = row[6]
            example['relationship'] = row[7]
            example['race'] = row[8]
            example['sex'] = row[9]
            example['capital-gain'] = row[10]
            example['capital-loss'] = row[11]
            example['hours-per-week'] = row[12]
            example['native-country'] = row[13] 
            if row[14] == '0':
                example['label'] = 'no'
            elif row[14] == '1':
                example['label'] = 'yes'
            examples.append(example)
    return examples

'''
generate the training example
example = [{'attr1': , 'attr2': , 'attr3': }, ...],
'''
def process_test_data(file_name):
    examples = []
    with open(file_name, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            example = {}
            example['age'] = row[1]
            example['workclass'] = row[2]
            example['fnlwgt'] = row[3]
            example['education'] = row[4]
            example['education-num'] = row[5]
            example['marital-status'] = row[6]
            example['occupation'] = row[7]
            example['relationship'] = row[8]
            example['race'] = row[9]
            example['sex'] = row[10]
            example['capital-gain'] = row[11]
            example['capital-loss'] = row[12]
            example['hours-per-week'] = row[13]
            example['native-country'] = row[14]
            examples.append(example)
    return examples



if __name__ == '__main__':
    missing_data = {}
    continuous_to_category_data = {}
    train_examples = process_training_data('./Data/train_final.csv')
    missing_data = get_missing_data(train_examples)
    continuous_to_category_data = get_continuous_to_category_data()
    train_examples = update_missing_data(missing_data, train_examples)
    train_examples = continuous_to_category(continuous_to_category_data, train_examples)
    test_examples = process_test_data('./Data/test_final.csv')
    test_examples = update_missing_data(missing_data, test_examples)
    test_examples = continuous_to_category(continuous_to_category_data, test_examples)



    # # process training dataset, transfrom numerial attributes
    # train_examples = process_data('./data/bank/train.csv')
    # numerical_media_map = find_medias(train_examples)
    # train_examples = num_to_binary(numerical_media_map, train_examples)
    
    # # process test dataset, transfrom numerial attributes
    # test_examples = process_data('./data/bank/test.csv')
    # test_examples = num_to_binary(numerical_media_map, test_examples)

    # attributes; big than or equal to the media =>yes, else no
    attrs = {
            'age': ['0a', '1a', '2a', '3a'],
            'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
            'fnlwgt': ['0a', '1a', '2a', '3a'],
            'education': ['Bachelors', 'Some-college', '11th','HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
            'education-num': ['0a', '1a', '2a', '3a'],
            'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
            'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
            'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
            'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
            'sex': ['Female', 'Male'],
            'capital-gain': ['0a', '1a'],
            'capital-loss': ['0a', '1a'],
            'hours-per-week': ['0a', '1a', '2a'],
            'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 
            'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
        }
    
    # boosting algorithm setting
    T = 500   # the number of iterations

    # get the final hypothesis
    boosting = Boosting(train_examples, test_examples, attrs, T)
    final_hypothesis = boosting.get_final_hypothesis()

    # do the test
    print('final result')
    #print(final_hypothesis)
    print(boosting.training_error)
    
    trainError = []
    for error_ in boosting.training_error:
        trainError.append([error_])

    with open('./Results/trainError.csv', 'w') as f:
        csv_f = csv.writer(f, lineterminator='\n')
        csv_f.writerows(trainError)



    # print(boosting.training_current_error)
    # print(boosting.test_error)
    # print(boosting.test_current_error)

    # generate the result files
    # boosting.store_test_result()


''''
Boosting 算法的处理流程：

1.对训练数据和测试数据进行预处理，每个item表示为一个dict，由键值对组成的，尤其的训练数据有一个weight权重，代表该点的位置
2.将训练数据作为算法的输入，开始训练boosting算法
    1. 确定好训练的循环次数（弱分类器的个数）
    2. 针对于每个循环
        1. 根据information gain选择最佳的feature来分割
        2. 确定这个分类器的分类规则，（使用哪个属性进行分割，那个属性值的输出是yes，哪个属性值的输出是no）
        3. 根据这个分类器，测试一下在训练集上的错误率。
        4. 有了错误率，计算这个弱分类的vote
        5. 根据以上两个重新更新每个item的权重。
        6. 注意每个item都是有自己的权重的。
    3. 生成最终的假设, 【假设1， 假设2， 。。。】其中假设1 =【vote， 选中的属性，其中label为yes的属性值】
3.数据测试
    1. 根据最终的假设，计算出预测值
    2. 结合预测值和真实值，看是否预测准确了。

作业：
（1）计算训练误差和测试误差，其中训练的迭代次数为从1-500

（2）计算训练周期为500的那一次的每一个的训练误差
'''