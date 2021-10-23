import csv
import copy
import math
from os import O_CLOEXEC
import random
import numpy as np


class ID3Solver:
    most_label_value = ''
    gain_method = ''
    max_depth = ''

    def __init__(self, label, gain_method, max_depth):
        self.most_label_value = label
        self.gain_method = gain_method
        self.max_depth = max_depth
        
    # calculate the information gain
    def get_gain(self, examples, attr, values):
        # get the proportions of different label values
        def get_label_prop_dict(sub_examples):
            res = {}
            for example in sub_examples:
                if example['label'] in res.keys():
                    res[example['label']] += 1
                else:
                    res[example['label']] = 1
            for key in res:
                res[key] /= len(sub_examples)
            return res

        def Entropy(sub_examples):
            res = 0
            label_prop_dict = get_label_prop_dict(sub_examples)
            for key in label_prop_dict:
                val = label_prop_dict[key]
                res -= val * math.log(val, 2)
            return res

        def ME(sub_examples):
            if len(sub_examples) == 0:
                return 0
            else:
                label_prop_dict = get_label_prop_dict(sub_examples)
                vals = label_prop_dict.values()
                return 1-max(vals)
            
        def GI(sub_examples):
            res = 1
            label_prop_dict = get_label_prop_dict(sub_examples)
            for key in label_prop_dict:
                val = label_prop_dict[key]
                res -= val*val
            return res
        
        def get_purity(examples):
            if self.gain_method == 0:
                return Entropy(examples)
            elif self.gain_method == 1:
                return ME(examples)
            elif self.gain_method == 2:
                return GI(examples)
            else:
                return Entropy(examples)

        # purity: Examples
        purity_examples = get_purity(examples)

        # purity: subExamples
        purity_values = []
        proportions = []
        num_Examples = len(examples)
        for value in values:
            sub_examples = []
            for example in examples:
                if example[attr] == value:
                    sub_examples.append(example)               
            purity_values.append(get_purity(sub_examples))
            proportions.append(len(sub_examples)/num_Examples)
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

    # get the most common feature
    def get_most_common_label(self, examples):
        label_values = ['yes', 'no']
        # get the most common label
        label_num = [0, 0]
        for example in examples:
            lb = example['label']
            if lb == 'yes':
                label_num[0] += 1
            elif lb == 'no':
                label_num[1] += 1
        most_label_value = label_values[label_num.index(max(label_num))]

        return most_label_value

    # run ID3, genearte the decision tree
    def ID3(self, examples, attrs, node):
        # check it is the max_depth, make it a leaf
        if node['depth'] == self.max_depth:
            node['is_leaf'] = True
            node['label'] = self.get_most_common_label(examples)
            return
        if len(attrs) == 0:
            print(examples)
        
        # find the best attr to split the examples, then add a root node to this tree
        attr = self.get_best_attr(examples, attrs)           # the selected attr

        sub_attrs = copy.deepcopy(attrs)
        del sub_attrs[attr]
        # add node to the structure
        node['feature'] = attr
        node['values'] = {}
        node['is_leaf'] = False
        
        # for each value of the attr, get a sub_example, and generate a branch for it
        for attr_v in attrs[attr]:
            # add a branch
            node['values'][attr_v] = {}
            cur_node = node['values'][attr_v]
            cur_node['depth'] = node['depth']+1

            sub_examples = self.get_sub_examples(examples, attr, attr_v)

            # if the subset is empty, then make the subset as a leaf, and set the most common label
            if len(sub_examples) == 0:
                # add a leaf node
                cur_node['is_leaf'] = True
                cur_node['label'] = self.get_most_common_label(examples)
                
            # else if the subset has the same label, make the subset as a leaf and set the common label
            elif self.has_same_label(sub_examples):
                cur_node['is_leaf'] = True
                cur_node['label'] = sub_examples[0]['label']

            # else if the subset has different labels, then run ID3() recursively  
            else:
                self.ID3(sub_examples, sub_attrs, cur_node)


class Training:
    gain_method = ''
    max_depth = ''
    examples = ''

    def __init__(self, gain_method, max_depth, examples):
        self.gain_method = gain_method
        self.max_depth = max_depth
        self.examples = examples

    def get_most_common_label(self):
        label_values = ['yes', 'no']

        #get the most common label
        label_num = [0, 0]
        for example in self.examples:
            lb = example['label']
            if lb == 'yes':
                label_num[0] += 1
            elif lb == 'no':
                label_num[1] += 1

        most_label_value = label_values[label_num.index(max(label_num))]
        return most_label_value

    def train(self):
        # big than or equal to the media =>yes, else no
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

        # get the most common label
        most_label_value = self.get_most_common_label()

        decision_tree = {'depth': 0}

        ID3_obj = ID3Solver(most_label_value, self.gain_method, self.max_depth)
        ID3_obj.ID3(self.examples, attrs, decision_tree)

        return decision_tree

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

# test the data
def test_data(tree, test_examples):
    num = len(test_examples)
    correct_num = 0

    for example in test_examples:
        cur_node = tree
        while not cur_node['is_leaf']:
            cur_node = cur_node['values'][example[cur_node['feature']]]
        predict_label = cur_node['label']
        real_label = example['label']
        if predict_label == real_label:
            correct_num += 1
    
    return 1-correct_num/num

# return the sampled training examples
def get_sample_train_examples(train_examples, num):
    sample_train_examples = []
    sample_train_examples_id = []
    length = len(train_examples)

    while len(sample_train_examples_id) < num:
        selected_id = random.randint(0, length-1)
        if selected_id not in sample_train_examples_id:
            sample_train_examples_id.append(selected_id)
    for id in sample_train_examples_id:
        sample_train_examples.append(train_examples[id])

    return sample_train_examples

# predict the result of examples
def predict_reult(tree, examples, result_lst):
    num = len(examples)
    for i in range(num):
        example = examples[i]
        cur_node = tree
        while not cur_node['is_leaf']:
            cur_node = cur_node['values'][example[cur_node['feature']]]
        predict_label = cur_node['label']
        if predict_label == 'yes':
            result_lst[i][0] += 1
        elif predict_label == 'no':
            result_lst[i][1] += 1
    return result_lst

# predict the error rate in terms of examples and the accumulate results
def get_current_error(examples, result_lst):
    num = len(examples)
    correct_num = 0
    for i in range(num):
        example = examples[i]
        real_label = example['label']
        predict_label = ''
        if result_lst[i][0] > result_lst[i][1]:
            predict_label = 'yes'
        else:
            predict_label = 'no'
        if real_label == predict_label:
            correct_num += 1  
    return 1 - correct_num/num   



if __name__ == '__main__':
    '''
    gain method: 0-entropy, 1-ME, 2-GI
    '''
    train_examples = process_data('./data/bank/train.csv')
    numerical_media_map = find_medias(train_examples)
    train_examples = num_to_binary(numerical_media_map, train_examples)
    
    test_examples = process_data('./data/bank/test.csv')
    test_examples = num_to_binary(numerical_media_map, test_examples)

    print('Proccessing dataset part end!')

    train_num = len(train_examples)
    test_num = len(test_examples)
    tree_num = 500      # each bagged trees has 500 trees
    bagged_trees_num = 100      # 100 bagged trees
    sample_num = 1000

    single_trees = []       # the collection of all single trees
    all_test_example_result = []        # all of predicted results of 100 bagged trees
    for i in range(test_num):
        all_test_example_result.append([0, 0])  # [the number of yes, the number of no]

    # iterate 100 bagged trees
    for bt in range(bagged_trees_num):
        test_example_result = []
        # initialize the result of training and test results respectively
        for i in range(test_num):
            test_example_result.append([0, 0]) # [the number of yes, the number of no]
            
        # iterate 500 trees
        for i in range(tree_num):
            # get the sample training data 
            sample_train_examples = get_sample_train_examples(train_examples, sample_num)
            # get the decision tree for the data
            training_obj = Training(0, 16, sample_train_examples)
            tree = training_obj.train()
            if i == 0:
                single_trees.append(tree)
            # using the decision tree to predict the results of training examples and test examples
            test_example_result = predict_reult(tree, test_examples, test_example_result)
            # print('round:', i)
            # print(test_example_result[0:10])
            # print('-----------------------')
        
        # predict the current result
        for i in range(test_num):
            res = ''
            if test_example_result[i][0] > test_example_result[i][1]:
                res = 'yes'
                all_test_example_result[i][0] += 1
            else:
                res = 'no'
                all_test_example_result[i][1] += 1
        
        print('the bagged trees:', bt)
        
    # calculate the average result yes = 1, no = 0 (all bagged trees)
    bagged_trees_average_res = []
    for i in range(test_num):
        bagged_trees_average_res.append(all_test_example_result[i][0]*1/bagged_trees_num)

    # compute the bias term and variance term of each test example
    bagged_trees_bias_term_lst = []
    bagged_trees_variance_term_lst = []
    for i in range(test_num):
        test_real_res = ''
        if test_examples[i]['label'] == 'yes':
            test_real_res = 1
        elif test_examples[i]['label'] == 'no':
            test_real_res = 0
        bagged_trees_bias_term_lst.append(math.pow(bagged_trees_average_res[i]-test_real_res, 2))

        variance = (math.pow(bagged_trees_average_res[i]-1, 2)*all_test_example_result[i][0] 
        + math.pow(bagged_trees_average_res[i]-0, 2)*all_test_example_result[i][1])/(bagged_trees_num-1)
        bagged_trees_variance_term_lst.append(variance)
    
    print('bagged_trees_bias_term_lst', bagged_trees_bias_term_lst)
    print('bagged_trees_variance_term_lst', bagged_trees_variance_term_lst)
    
    # get the avg of bias and variance of all the test examples
    bagged_trees_avg_bias = 0
    bagged_trees_avg_variance = 0
    for i in range(test_num):
        bagged_trees_avg_bias += bagged_trees_bias_term_lst[i]
        bagged_trees_avg_variance += bagged_trees_variance_term_lst[i]
    bagged_trees_avg_bias /= test_num
    bagged_trees_avg_variance /= test_num

    print('bagged_trees_avg_bias', bagged_trees_avg_bias)
    print('bagged_trees_avg_variance', bagged_trees_avg_variance)


    # -----------------------------------------------Single tree Part------------------------------------------------------------------

    single_all_test_example_result = []        # all of predicted results of 100 single trees
    for i in range(test_num):
        single_all_test_example_result.append([0, 0])  # [the number of yes, the number of no]
    
    for tree in single_trees:
        single_all_test_example_result = predict_reult(tree, test_examples, single_all_test_example_result)
    
    # calculate the average result yes = 1, no = 0 (all bagged trees)
    single_trees_average_res = []
    for i in range(test_num):
        single_trees_average_res.append(single_all_test_example_result[i][0]*1/bagged_trees_num)
    
    # compute the bias term and variance term of each test example
    single_trees_bias_term_lst = []
    single_trees_variance_term_lst = []
    for i in range(test_num):
        test_real_res = ''
        if test_examples[i]['label'] == 'yes':
            test_real_res = 1
        elif test_examples[i]['label'] == 'no':
            test_real_res = 0
        single_trees_bias_term_lst.append(math.pow(single_trees_average_res[i]-test_real_res, 2))

        variance = (math.pow(single_trees_average_res[i]-1, 2)*single_all_test_example_result[i][0] 
        + math.pow(single_trees_average_res[i]-0, 2)*single_all_test_example_result[i][1])/(bagged_trees_num-1)
        single_trees_variance_term_lst.append(variance)
    
    # get the avg of bias and variance of all the test examples
    single_trees_avg_bias = 0
    single_trees_avg_variance = 0
    for i in range(test_num):
        single_trees_avg_bias += single_trees_bias_term_lst[i]
        single_trees_avg_variance += single_trees_variance_term_lst[i]
    single_trees_avg_bias /= test_num
    single_trees_avg_variance /= test_num

    print('single_trees_avg_bias', single_trees_avg_bias)
    print('single_trees_avg_variance', single_trees_avg_variance)


''''
1.生成100个bagged trees；其中每个bagged tree由500个tree组成（每个tree 不重叠地sample 1000个数据（注意树深度））

针对 the single tree learner
1.从这100个bagged trees中分别取出第一个tree。
2.针对每一个测试数据：
    1.分别计算出100个single tree的平均结果+测试数据的真实结果=》得到bias term
    2.使用这100个single tree的结果=》Sample variance
3.求所有的bias term和Sample variance的平均值

针对 the bagged trees
1.针对每一个测试数据：
    1.分别计算出100个bagged trees的平均结果+测试数据的真实结果=》得到bias term
    2.使用这100个bagged trees的结果=》Sample variance
2.求所有的bias term和Sample variance的平均值

代码思路：
1. all_test_res = [[num of yes, num of no], []..]  # for each bagged trees   single_trees = []
2. repeat 100 bagged trees
    1. test_res = [[num of yes, num of no], []]    # for each tree in bagged trees
    2. repeat 500 tree
        1. get 1000 sample data
        2. train the data and get the decision tree
        3. if this is the first tree, store it into the single_trees
        4. use the decision tree to predict the result (i.e. update the test_res)
    3. predict the result of test and update the all_test_res
3. based on all_test_res, calculate the average_result
'''