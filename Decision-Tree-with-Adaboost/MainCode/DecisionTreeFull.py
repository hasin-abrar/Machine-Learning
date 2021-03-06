# pre processing

import math
import random

import datetime
import numpy as np
import pandas as pd


class PreProcessing(object):
    def __init__(self, examples):
        self.examples = examples

    # takes a list as input and gives the mode. (1)[0][0] 1st appearance and more signifies
    def Most_Common(self, lst):
        from collections import Counter
        data = Counter(lst)
        return data.most_common(1)[0][0]

    # missing_col = Find_Missing_Col(examples,4)
    def Find_Missing_Col(self,attr_length):
        missing_col = []
        for i in range(attr_length):
            for e in examples:
                if e[i] == " ":
                    # print(e[i])
                    missing_col.append(i)
                elif isinstance(e[i], str):
                    continue
                # print(type(e[i]))
                # print(len(e[i]),e[i])
                if math.isnan(e[i]):
                    missing_col.append(i)
                    break
        return missing_col

    # number_type is a list
    # number_type = [1,2]
    # Replace_With_Mean(examples,number_type)

    def Replace_With_Mean(self, examples_df, examples, number_type):
        from sklearn.preprocessing import Imputer
        imputer = Imputer(missing_values='NaN',
                          strategy='mean', axis=0)
        for n in number_type:
            imputer = imputer.fit(examples[:, n:(n + 1)])
            examples[:, n:(n + 1)] = imputer.transform(examples[:, n:(n + 1)])


    # examples = Remove_Useless_Rows(examples)
    def Remove_Useless_Rows(self, examples):
        index = -1
        for e in examples:
            index += 1
            last_value = e[-1]
            # print(last_value,index)
            if isinstance(last_value, str):
                continue
            if math.isnan(last_value):
                # print(index)
                examples = np.delete(examples, index, 0)
                # print(examples)
        return examples

    def Replace_With_Mode(self, examples, string_type):
        for s in string_type:
            single_col = examples[:, s]
            _max_appearance = self.Most_Common(single_col)
            for j in range(len(single_col)):
                if isinstance(single_col[j], str):
                    continue
                if math.isnan(single_col[j]):
                    single_col[j] = _max_appearance

    def GetBooleanEntropy(self, yes, no):
        succ_prob = (yes / (yes + no))
        if succ_prob == 0:
            return 0
        elif succ_prob == 1:
            return 0
        # print ("succ : ",succ_prob)
        return -(succ_prob * math.log2(succ_prob) + (1 - succ_prob) * math.log2((1 - succ_prob)))

    def Get_Split_Val(self, examples, index):
        #    selected_col = examples
        sorted_col = sorted(examples, key=lambda k: k[index])
        
        print(sorted_col)
        start = sorted_col[0][index] - 10
        #    end = sorted_col[0] + 10
        class_col = examples[:, -1]
        yes = []
        no = []
        yes.append(0)
        yes.append(0)
        no.append(0)
        no.append(0)
        pos = neg = 0
        # print(class_col)
        for c in class_col:
            if c == "Yes":
                yes[1] += 1
            else:
                no[1] += 1
        pos = yes[1]
        neg = no[1]
        # print(yes, no)
        init_entropy = self.GetBooleanEntropy(yes[1], no[1])
        _max = 0
        split = start
        for j in range(len(sorted_col) - 1):
            mid = (sorted_col[j][index] + sorted_col[j + 1][index]) / 2
            remainder_attrb_entropy = 0
            if sorted_col[j][-1] == "Yes":
                yes[0] += 1
                yes[1] -= 1
            else:
                no[0] += 1
                no[1] -= 1
            for k in range(2):
                remainder_attrb_entropy += ((yes[k] + no[k]) / (pos + neg)) * self.GetBooleanEntropy(yes[k], no[k])
            gain = init_entropy - remainder_attrb_entropy

            if gain > _max:
                _max = gain
                split = mid
        # print(split)
        return split

    def Binarization(self, examples, num_type):
        for n in num_type:
            split_val = self.Get_Split_Val(examples, n)
            # print("##########")
            # print("attribute ", n, " : ", split_val)
            changed_col = examples[:, n]
            for i in range(len(changed_col)):
                if changed_col[i] <= split_val:
                    changed_col[i] = -1  # making all values having same type
                else:
                    changed_col[i] = +1

    def GetAttributeList(self, dataframe):
        attr_list = []
        for i in range(len(list(dataframe)) - 1):
            attr_list.append(i)
        return attr_list

    def GetAtrributeLength(self, dataframe):
        return len(dataframe.columns) - 1

    def GetAtrributeMapping(self,examples,attr_length):
        attr_mapping = {}
        index = {}
        for i in range(attr_length):
            single_col = examples[:,i]
            attr_types = list( set(single_col) )
            attr_name = i
            attr_mapping[attr_name] = attr_types
            index[attr_name] = i
        return attr_mapping, index

    def DoLastColEncoding(self,examples,last_col,choice):
        y = examples[:,-1]
        for i in range(len(examples)):
            y[i] = y[i].strip()
            if choice == 2:
                y[i] = y[i].strip('.')
            if y[i] == last_col[0]:
                y[i] = 1
            else:
                y[i] = -1
        return y

    def GetTrainTestSplit(self,x,y,split_size):
        # from sklearn.cross_validation import train_test_split
        from sklearn.model_selection import train_test_split
        return train_test_split(x, y,test_size=split_size,random_state=60)



class Node:
    def __init__(self,val,isLeaf):
        self.child = []
        self.val = val
        self.isLeaf = isLeaf

    # sutree is also a node
    def insert(self,subtree):
        self.child.append(subtree)

class DecisionTree:
    def __init__(self,attr_mapping,index,depth_max):
        self.attr_mapping = attr_mapping
        self.index = index
        self.depth_max = depth_max

    def setMaxDepth(self,depth):
        self.depth_max = depth

    def GetBooleanEntropy(self,yes,no):
        if (yes + no) == 0:
            # print("WHAT")
            return 0
        succ_prob = (yes / (yes + no))
        if succ_prob == 0:
            return 0
        elif succ_prob == 1:
            return 0
        else:
            # print ("succ : ",succ_prob)
            return -(succ_prob * math.log2(succ_prob) + (1 - succ_prob) * math.log2((1 - succ_prob)))

    # attribute is a String, index is an integer
    def Importance(self,attribute,x_train,y_train, index):
        yes = no = 0
        remainder_attrb_entropy = 0
        for y in y_train:
            if y == 1: #this means class "Yes"
                yes+=1
            else:
                no+=1
        attr_entropy = self.GetBooleanEntropy(yes,no)
        # all the attribute values of that attribute = list
        # attr_vals is a list
        attr_vals = self.attr_mapping[attribute]
        pos = []
        neg = []
        for j in range(len(attr_vals)):
            pos.append(0)
            neg.append(0)
        for i in range(len(x_train)):
            for j in range(len(attr_vals)):
                # example has the same attribute value
                if x_train[i][index] == attr_vals[j] :
                    if y_train[i] == 1:
                        pos[j] += 1
                    else:
                        neg[j] += 1
                    break

        for k in range(len(attr_vals)):
            weight = ((pos[k] + neg[k])/(yes+no) )
            # print(weight)
            remainder_attrb_entropy += weight* self.GetBooleanEntropy(pos[k],neg[k] )

        return attr_entropy - remainder_attrb_entropy

    # attributes is a list of attribute(String)
    def Dec_Tree_Learning(self,x_train,y_train,attributes,par_x_train,par_y_train,depth):
        same_class = 1
        yes = no = 0
        if depth >= self.depth_max:
            return self.Plurality_Value(y_train)
        for y in y_train:
            if y == 1:
                yes += 1
            else:
                no += 1
            if yes >0 and no >0:
                same_class = 0
                break

        if len(x_train) == 0:
            return self.Plurality_Value(par_y_train)
        elif same_class == 1:
            if yes > 0 :
                return Node(1,1)
            else :
                return Node(-1,1)
        elif len(attributes) == 0:
            return self.Plurality_Value(y_train)
        else:
            _max = -1
            root = attributes[0]
            for a in attributes: # 'a' is an int
                importance = self.Importance(a,x_train,y_train,self.index[a])
                if importance > _max:
                    _max = importance
                    root = a
            tree = Node(root,0)
            attribute_list = self.attr_mapping[root]
            for a in attribute_list: # each a is a attribute value
                child_x_train = []
                child_y_train = []
                for i in range(len(x_train)):
                    # attribute index and its corresponding value in example e[index[root]]
                    if x_train[i][self.index[root]] == a:
                        child_x_train.append(x_train[i])
                        child_y_train.append(y_train[i])
                new_attributes = []
                for a in attributes:
                    if a == root:
                        continue
                    new_attributes.append(a)
                subtree = self.Dec_Tree_Learning(child_x_train,child_y_train,new_attributes,x_train,y_train,depth+1)
                tree.insert(subtree)
            return tree



    def Plurality_Value(self,y_val):
        yes = no = 0
        for y in y_val:
            if y == 1:
                yes+=1
            else:
                no+=1
        if yes > no:
            return Node(1,1) # 1st 1 = Yes
        else:
            return Node(-1,1) # 1st 0 = No


    def Prediction(self,x_test, node):
        if node.isLeaf == 1 :
            return node.val
        attr = node.val
        attr_list = self.attr_mapping[attr]
        indx = self.index[attr]
        found = False
        next_node = Node(0,0)
        for i in range(len(attr_list)):
            if x_test[indx] == attr_list[i]:
                found = True
                next_node = node.child[i]
                break
        if found != True :
            print(indx," Default in Searching !",x_test)
            defaultNode = self.Plurality_Value(x_test)
            return defaultNode.val
        else:
            return self.Prediction(x_test,next_node)


    def Adaboost(self,x_train,y_train, k_count, attributes):
        h = []
        z = []
        weight = []
        x_train_index = []
        y_train_index = []
        for i in range(len(x_train)):
            weight.append((1 / len(x_train)))
            x_train_index.append(i)
            y_train_index.append(i)
        # print(weight)
        for k in range(k_count):
            z.append(0.0)
            node = Node(0,0)
            h.append(node)
            next_x_train = []
            next_y_train = []
            # data = examples_dataframe.sample(len(examples_dataframe), weights=weight)
            data = np.random.choice(x_train_index, len(x_train_index), p=weight)
            for ind in data:
                next_x_train.append(x_train[ind])
                next_y_train.append(y_train[ind])
            h[k] = self.Dec_Tree_Learning(next_x_train,next_y_train, attributes,[], [], 0)
            error = 0
            for j in range(len(x_train)):
                if self.Prediction( x_train[j],h[k]) != y_train[j]:
                    error += weight[j]
            if error > 0.5:
                k -= 1
                print("K KOMSEEEE")
                continue
            # print(k," Error : ",error)
            for j in range(len(x_train)):
                if self.Prediction( x_train[j],h[k]) == y_train[j]:
                    weight[j] *= (error / (1 - error))
            # weight = preprocessing.normalize(weight)
            weight = [ float(i) / sum(weight) for i in weight ]
            z[k] = math.log10((1 - error) / error)
        return Weighted_Majority(h,z)

    def Prediction_Stump(self,weighted_majority, k_count, x_test):
        val = 0
        h = weighted_majority.h
        z = weighted_majority.z
        for i in range(k_count):
            pred = self.Prediction( x_test,h[i])
            # print (pred, z[i])
            val += (  pred* z[i] )
        # print("final : ",val)
        if val > 0:
            return 1
        else:
            return -1

class Weighted_Majority:
    def __init__(self,h,z):
        self.z = z
        self.h = h


def PreProcessData(dataset_frame,number_type,replace_with_mean,dropping_col,last_col,choice):
    dataset_frame = dataset_frame.replace(' ', np.NaN)
    dataset_frame = dataset_frame.replace('?', np.NaN)
    dataset_frame.drop(dataset_frame.columns[dropping_col], axis=1, inplace=True)

    examples = dataset_frame.iloc[:, :].values
    examples_df = dataset_frame.iloc[:, :]
    # x = dataset_frame.iloc[:, :-1].values
    # y = dataset_frame.iloc[:, -1].values
    if choice == 0:
        extra_sample_count = 7
        examples_filtered = []
        examples_filtered_index = []
        for i in range(len(examples)):
            if examples[i][-1] == 1:
                examples_filtered.append(list(examples[i]) )
                examples_filtered_index.append(i)
        indx_count = 0
        while(True):
            rand_index = random.randint(0,len(examples) - 1)
            if rand_index not in examples_filtered_index:
                examples_filtered.append(list(examples[rand_index]) )
                indx_count += 1
            if indx_count == extra_sample_count:
                break
        random.shuffle(examples_filtered)
        print(examples)
        examples_filtered = np.array(examples_filtered)
        print("########### FILTERED ##############")
        print(examples_filtered)
        examples = examples_filtered

    pre_processing = PreProcessing(examples)
    attr_list = pre_processing.GetAttributeList(examples_df)
    # attr_length does not include label (last col)
    attr_length = pre_processing.GetAtrributeLength(dataset_frame)
    replace_with_mode = []
    for i in range(attr_length):
        if i in number_type:
            continue
        replace_with_mode.append(i)

    examples = pre_processing.Remove_Useless_Rows(examples)
    pre_processing.Replace_With_Mean(examples_df, examples, replace_with_mean)
    pre_processing.Replace_With_Mode(examples, replace_with_mode)

    for n in number_type:
        examples[:,n:(n+1)] = examples[:,n:(n+1)].astype(np.float64)

    pre_processing.Binarization(examples, number_type)

    attr_mapping, index = pre_processing.GetAtrributeMapping(examples, attr_length)

    y = pre_processing.DoLastColEncoding(examples, last_col,choice)
    x = examples[:, :-1]

    return x, y, attr_mapping, index, attr_list,pre_processing


##################### START #######################

choice = 0
dropping_col = []
number_type = []

dataset_test = []
if choice == 0:
    dataset = pd.read_csv("Data.csv")
    last_col = ["1", "0"]
    dropping_col = []
    replace_with_mean = number_type = [1, 2]
elif choice == 1:
    dataset = pd.read_csv("1/1.csv")
    last_col = ["Yes", "No"]
    dropping_col = [0]
    replace_with_mean = number_type = [4, 17, 18]
elif choice == 2:
    dataset = pd.read_csv("2/2.csv",header = None)
    dataset_test = pd.read_csv("2/2_test.csv",header = None)
    last_col = ["<=50K", ">50K"]
    dropping_col = []
    number_type = [0, 2, 4, 10, 11, 12]
    replace_with_mean = [0,2,4,12]
elif choice == 3:
    dataset = pd.read_csv("3/3.csv")

    last_col = ["1", "0"]
    dropping_col = []
    attrb_length = len(dataset.columns) - 1
    for i in range(attrb_length):
        number_type.append(i)
    # number_type = [0, 2, 4, 10, 11, 12]
    print(number_type,len(number_type))
    replace_with_mean = number_type
else:
    dataset = pd.read_csv("Data.csv")
    last_col = ["Yes", "No"]
    dropping_col = []
    replace_with_mean = number_type = [1, 2]

x, y, attr_mapping, index, attr_list,pre_processing = PreProcessData(dataset,number_type, replace_with_mean,dropping_col,last_col,choice)
'''
dataset = dataset.replace(' ',np.NaN)
dataset = dataset.replace('?',np.NaN)
dataset.drop(dataset.columns[dropping_col], axis=1, inplace=True)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
examples = dataset.iloc[:, :].values
examples_df = dataset.iloc[:, :]

pre_processing = PreProcessing(examples)
attr_list = pre_processing.GetAttributeList(examples_df)
# attr_length does not include label (last col)
attr_length = pre_processing.GetAtrributeLength(dataset)

for i in range(attr_length):
    if i in number_type:
        continue
    string_type.append(i)

# missing_col = pre_processing.Find_Missing_Col(attr_length)
# print(missing_col)
examples =  pre_processing.Remove_Useless_Rows(examples)
pre_processing.Replace_With_Mean(examples_df,examples,number_type)
pre_processing.Replace_With_Mode(examples,string_type)
pre_processing.Binarization(examples,number_type)

attr_mapping, index = pre_processing.GetAtrributeMapping(examples,attr_length)

y = pre_processing.DoLastColEncoding(examples,last_col)
x = examples[:,:-1]
'''


if choice == 2:
    x_train = x
    y_train = y
    x_test, y_test,*rest = PreProcessData(dataset_test, number_type, replace_with_mean, dropping_col, last_col,choice)

else:
    x_train, x_test, y_train, y_test = pre_processing.GetTrainTestSplit(x,y,split_size=0.2)
# print(x_train,"\n",y_train,"\n", x_test ,"\n", y_test)

# print(examples)

############### Decision Tree Learning ############

decision_tree = DecisionTree(attr_mapping,index, math.inf)
decision_tree_adaboost = DecisionTree(attr_mapping,index,1)
print(len(x_train), len(x_test))

tree = decision_tree.Dec_Tree_Learning(x_train,y_train,attr_list,[],[],0)
not_match = match = 0
# print(tree)
for i in range(len(x_test)):
    # y_test[i] = y_test[i].strip()
    # if choice == 2:
    #     y_test[i] = y_test[i].strip('.')
    if decision_tree.Prediction(x_test[i],tree) == y_test[i]:
        match += 1
        # print("Match")
    else:
        not_match += 1
        # print("Does not match")

print(match,not_match)
accuracy = (match)/ (match + not_match) * 100
print("Decision Tree : ",accuracy,"%","Time : ",datetime.datetime.now().time())


print("*******Adaboost*******")
k_list = [5,10,15,20]
for k in k_list:
    k_count = k
    weighted_majority = decision_tree_adaboost.Adaboost(x_train,y_train,k_count,attr_list)
    not_match = match = 0
    for i in range(len(x_test)):

        if decision_tree_adaboost.Prediction_Stump(weighted_majority,k_count,x_test[i]) == y_test[i]:
            match += 1
            # print("Match")
        else:
            not_match += 1
            # print("Does not match")

    accuracy = (match)/ (match + not_match) * 100
    print("LoopCount : ",k," accuracy : ",accuracy,"%","Time : ",datetime.datetime.now().time())

# '''