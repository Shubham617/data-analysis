import numpy as np
from Data import Data

"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because 
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""

class Node(object):
    def __init__(self, att_idx=None, att_values=None, answer=None):
        self.att_idx = att_idx
        self.att_values = att_values
        self.branches = {}
        self.answer = answer

    def route(self, sample):
        if len(self.branches) < 1:
            return self.answer
        att = sample[self.att_idx]
        return self.branches[att].route(sample)

class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, examples, attribute_names, attribute_values):
        """
        :param examples: input data, a list (len N) of lists (len num attributes)
        :param attribute_names: name of each attribute, list (len num attributes)
        :param attribute_values: possible values for each attribute, a list (len num attributes) of lists (len num values for each attribute)
        """
        self.attribute_names = attribute_names
        self.attribute_values = attribute_values
        self.attribute_idxs = dict(zip(attribute_names, range(len(attribute_names))))
        self.attribute_map = dict(zip(attribute_names, attribute_values))
        self.P = sum([example[-1] for example in examples])
        self.N = len(examples) - self.P
        self.H_Data = self.H(self.P/(self.P+self.N))  # this is B on p.704 - to be used in InfoGain
        self.root = self.DTL(examples, attribute_names)

    def same_classfication(self, examples):
        allTrue = 0
        allFalse = 0

        for example in examples:
            if example[len(example)-1] == True:
                allTrue = 1
            if example[len(example) - 1] == False:
                allFalse = 1

        if(allTrue == 1 and allFalse == 1):
            return False
        return True

    def DTL(self, examples, attribute_names, default=True):
        """
        Learn the decision tree.
        :param examples: input data, a list (len N) of lists (len num attributes)
        :param attribute_names: name of each attribute, list (len num attributes)
        :return: the root node of the decision tree
        """
        # WRITE the required CODE HERE and return the computed values

        import copy

        if len(examples)==0:
            return default
        elif self.same_classfication(examples):
            return examples[0][len(examples[0])-1]
        elif len(attribute_names) == 0:
            return self.mode(examples)

        best = self.chooseAttribute(attribute_names, examples)

        tree = Node(att_idx=self.attribute_idxs[best], att_values=self.attribute_map[best])


        value_list = self.attribute_map[best]
        for value in value_list:
            examples_i = []
            index = self.attribute_idxs[best]
            for example in examples:
                if example[index] == value:
                    examples_i.append(example)
            attribute_names_copy = copy.deepcopy(attribute_names)
            attribute_names_copy.remove(best)
            subtree = self.DTL(examples_i, attribute_names_copy,self.mode(examples))
            if subtree == True or subtree == False:
                tree.branches[value] = Node(answer=subtree)
            else:
                tree.branches[value] = subtree

        return tree

    def mode(self, answers):
        """
        Compute the mode of a list of True/False values.
        :param answers: a list of boolean values
        :return: the mode, i.e., True or False
        """
        # WRITE the required CODE HERE and return the computed values
        true_ctr = 0
        false_ctr = 0

        for answer in answers:
            if(answer[len(answer)-1]) == True:
                true_ctr += 1
            if(answer[len(answer)-1]) == False:
                false_ctr += 1

        return True if (true_ctr >= false_ctr) else False

    def H(self,p):
        import math
        """
        Compute the entropy of a binary distribution.
        :param p: p, the probability of a positive sample
        :return: the entropy (float)
        """
        # WRITE the required CODE HERE and return the computed values
        #print("probability", p)
        if p == 0.0 or p == 1.0:
            return 0.0
        return -(p*math.log(p,2)+(1-p)*math.log((1-p),2))

    def ExpectedH(self, attribute_name, examples):
        """
        Compute the expected entropy of an attribute over its values (branches).
        :param attribute_name: name of the attribute, a string
        :param examples: input data, a list of lists (len num attributes)
        :return: the expected entropy (float)
        """
        # WRITE the required CODE HERE and return the computed values
        value_list = self.attribute_map[attribute_name]
        total_len = len(examples)
        #print("total length", total_len)
        index = self.attribute_idxs[attribute_name]
        sum = 0.0
        for val in value_list:
            p_k = 0
            n_k = 0
            for example in examples:
                if example[index] == val:
                    if example[len(example)-1] == True:
                        p_k += 1
                    if example[len(example) - 1] == False:
                        n_k += 1

            if p_k+n_k != 0:
                sum += ( ((p_k+n_k)/total_len)*self.H(p_k/(p_k+n_k)) )

        return sum

    def InfoGain(self, attribute_name, examples):
        """
        Compute the information gained by selecting the attribute.
        :param attribute_name: name of the attribute, a string
        :param examples: input data, a list of lists (len num attributes)
        :return: the information gain (float)
        """
        return self.H_Data - self.ExpectedH(attribute_name,examples)

    def chooseAttribute(self, attribute_names, examples):
        """
        Choose to split on the attribute with the highest expected information gain.
        :param attribute_names: name of each attribute, list (len num attributes)
        :param examples: input data, a list of lists (len num attributes)
        :return: the name of the selected attribute, string
        """
        InfoGains = []
        for att in attribute_names:
            InfoGains += [self.InfoGain(att,examples)]
        return attribute_names[np.argmax(InfoGains)]

    def predict(self, X):
        """
        Return your predictions
        :param X: inputs, shape:(N,num_attributes)
        :return: predictions, shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        # prediction should be a simple matter of recursively routing
        # a sample starting at the root node
        predictions = []
        for x in X:
            predictions.append(self.root.route(x))
        return np.asarray(predictions)

    def print(self):
        """
        Print the decision tree in a readable format.
        """
        self.print_tree(self.root)

    def print_tree(self,node):
        """
        Print the subtree given by node in a readable format.
        :param node: the root of the subtree to be printed
        """
        if len(node.branches) < 1:
            print('\t\tanswer',node.answer)
        else:

            att_name = self.attribute_names[node.att_idx]
            for value, branch in node.branches.items():
                print('att_name',att_name,'\tbranch_value',value)
                self.print_tree(branch)


if __name__ == '__main__':
    # Get data
    data = Data()
    examples, attribute_names, attribute_values = data.get_decision_tree_data()
    #print(examples)
    # Decision tree trained with max info gain for choosing attributes
    model = DecisionTree()
    model.fit(examples, attribute_names, attribute_values)
    print(examples)
    y = model.predict(examples)
    print(y)
    model.print()
