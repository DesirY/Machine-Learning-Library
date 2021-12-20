import csv
import copy
import math
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

# store the result
def store_test_result(tree, test_examples):
    num = 1
    result = [['ID', 'Prediction']]

    for example in test_examples:
        cur_node = tree
        while not cur_node['is_leaf']:
            cur_node = cur_node['values'][example[cur_node['feature']]]
        predict_label = cur_node['label']
        if predict_label == 'yes':
            result.append([num, 1])
        elif predict_label == 'no':
            result.append([num, 0])
        num += 1
    
    with open('./Results/result.csv', 'w') as f:
        csv_f = csv.writer(f, lineterminator='\n')
        csv_f.writerows(result)

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
    'capital-gain': [0], 
    'capital-loss': [0], 
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


if __name__ == '__main__':
    '''
    gain method: 0-entropy, 1-ME, 2-GI
    '''
    missing_data = {}
    continuous_to_category_data = {}
    train_examples = process_training_data('./Data/train_final.csv')
    # print(train_examples)
    missing_data = get_missing_data(train_examples)
    continuous_to_category_data = get_continuous_to_category_data()
    # print(continuous_to_category_data)
    # print(missing_data)
    # update missing data
    train_examples = update_missing_data(missing_data, train_examples)
    # print(train_examples)
    train_examples = continuous_to_category(continuous_to_category_data, train_examples)
    # print(train_examples)

    # numerical_media_map = find_medias(train_examples)
    # train_examples = num_to_binary(numerical_media_map, train_examples)
    
    test_examples = process_test_data('./Data/test_final.csv')
    test_examples = update_missing_data(missing_data, test_examples)
    test_examples = continuous_to_category(continuous_to_category_data, test_examples)

    # test_examples = num_to_binary(numerical_media_map, test_examples)

    # max_depth = ''

    print('Information Gain')
    print('--------------------------------------------')
    print('MaxDepth\tTrainingError\tTestError')
    for i in range(14):
        max_depth = i+1
        training_obj = Training(0, max_depth, train_examples)
        tree = training_obj.train()
        # print(tree)
        train_error = test_data(tree, train_examples)
        print(train_error)
        # test_error = test_data(tree, test_examples)
        # print((i+1), '\t', train_error, '\t', test_error)
        store_test_result(tree, test_examples)