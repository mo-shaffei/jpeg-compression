from operator import itemgetter
import math
import json


class Node:
    '''Class to represent a node in the huffman tree.'''
    
    def __init__(self, right = None, left = None):
        #right, and left are pointing to the child nodes,
        #and probability is used by the huffman algorithm
        self.right = right
        self.left = left
    
    @classmethod
    def make_parent(cls, right, left):
        '''create a parent for the two child nodes right and left. The probability stored
        in the parent is the summation of the probability in each child'''
        parent = Node(right, left)
        parent.probability = right.probability + left.probability
        return parent


class Leaf(Node):
    '''Leaf class inherits from Node class. Its purpose is to make the leaves
    special nodes that store an extra member which is the symbol.'''
    
    def __init__(self, symbol, probability):
        #use the parent constructor and init right and left to None since this is a leaf node
        super().__init__(None, None)
        self.symbol = symbol
        self.probability = probability


class Huffman:
    '''Class to implement huffman compression functionalities'''
    
    def encode_sequence(self, sequence, probabilities = None):
        '''encodes the input sequence, using the specified symbols probabilities dict.
        If probabilities dict is not given then the function will generate it for the given sequence.
        Returns the the encoded sequence, code dict, and a dict containing figures(H, L_avg, Efficiency)'''
        if not probabilities:
            #if dictionary of symbols and probabilities was not passed then create it
            probabilities = self._calculate_probabilities(sequence)
        #construct the huffman code for this symbols-probabilities combination
        code_dict = self.make_code(probabilities)
        #init the encoded_seq to the original sequence
        encoded_seq = sequence
        for symbol in probabilities:
            #replace each symbol by its code
            encoded_seq = [code_dict[symbol] if s == symbol else s for s in encoded_seq]
        return ''.join(encoded_seq), code_dict, self.get_figures(code_dict, probabilities)
    
    def encode_file(self, filename, output = "output.txt", probabilities = None):
        '''encodes the input file using the probabilities given in probabilities dict.
        If probabilities dict is not given then the function will genrate it for the given file.
        A new file containing the encoded sequence is stored. The file has name specified by the output parameter.
        A new JSON file containing the code is stored as well.
        Returns the code dict, and a dict containing figures(H, L_avg, Efficiency)'''
        
        # open file and read all its content into string sequence
        with open(filename, 'r') as file:
            sequence = file.read()
        
        # encode the string and get the encoded sequence and the code dictionary
        encoded_seq, code_dict, figures = self.encode_sequence(sequence, probabilities=probabilities)
        
        # write the encoded sequnce to the output file
        with open(output, 'w') as out_file:
            out_file.write(encoded_seq)
        
        # write the code dictionary to a json file
        with open("dictionary.json", 'w') as file:
            file.write(json.dumps(code_dict))
            
        return code_dict, encoded_seq, figures    
    
    def decode_sequence(self, sequence, code_dict):
        '''use the code_dict to decode the given sequence. Returns the decoded string'''
        
        #first use the code_dicionary to construct the huffman tree 
        root = self._construct_tree(code_dict)
        decoded_seq = []
        index = 0
        #keep looping as long as we did not reach the end of the sequence
        while index != len(sequence):
            #current_node will be used to iterate over nodes, and it is init to the root of the tree
            current_node = root
            #keep traversing the tree as long as we did not encounter a leaf
            while current_node.left or current_node.right:
                if sequence[index] == '1':
                    #if the current bit is 1 go the left child in the tree
                    current_node = current_node.left
                else:
                    #else the current bit is 0 go the left child in the tree
                    current_node = current_node.right
                #increment index to point to the next bit in the sequence
                index += 1
            #after the loop terminates, current_node will point to the leaf containing the symbol
            #so add it to the decoded sequence
            decoded_seq.append(current_node.symbol)
        return decoded_seq
    
    def decode_file(self, filename, output = "decoded.txt", code_dict = None):
        '''use the given code dictionry to decode the given file. if code_dictionary was not
        passed it will be loaded from a json file called dictionary.json. Return the decoded sequence.'''
        
        #read the encoded sequence into a string
        with open(filename, 'r') as file:
            sequence = file.read()
        
        #if code_dict was not passed to function then read it from json file
        if not code_dict:
            with open('dictionary.json', 'r') as file:
                code_dict = json.load(file) # use `json.loads` to do the reverse
        
        #decode the sequence using the code_dictcode
        decoded_seq = self.decode_sequence(sequence, code_dict)
        with open(output, 'w') as file:
            file.write(decoded_seq)
        
    def make_code(self, probabilities):
        '''encode the symbols using their probabilities given in the probabilities dictionary.
        Return a dictionary of symbols and their binary code'''
        
        #create a list of leaves/nodes from each symbol and its probability
        node_list = [Leaf(sym, prob) for sym, prob in probabilities.items()]
        #sort the list descendingly according to the probability in each node 
        node_list.sort(key=lambda x: x.probability, reverse=True)
        #keep looping as long as there are more than one node in the list
        while len(node_list) != 1:
            #pop nodes with least 2 probabilities from the list
            last1 = node_list.pop()
            last2 = node_list.pop()
            #make a single parent out of these nodes and append it to the list
            node_list.append(Node.make_parent(last2, last1))
            #sort the list to ensure probabilities are always sorted descendingly
            self._insert_sort(node_list)
        #init an empty dict which will store symbols and their code
        symbols_encoded = {}
        #extract leaves and their code from the huffman tree and populate the symbols_encoded dict
        self._extract_leaves(node_list[0], symbols_encoded)
        return symbols_encoded
    
    def get_figures(self, code_dict, probabilities):
        avg = 0
        for symbol, code in code_dict.items():
            avg += len(code)*probabilities[symbol]
        
        entropy = 0
        for _, prob in probabilities.items():
            entropy += -prob*math.log2(prob)
        
        efficiency = entropy / avg
        
        return {"h":entropy, "l":avg, "e":efficiency}
    
    def _construct_tree(self, code_dict):
        '''Takes a dictionary of symbols and their codes, and constructs the huffman tree
        of this code. Returns the root of the tree'''
        
        #create a root for the tree
        root = Node(None, None)
        #loop over all symbols and codes in the dictionary
        for sym, code in code_dict.items():
            #use current_node to iterate over nodes and init it to the root of the tree
            current_node = root
            #loop over all the bits of the current symbol except the last bit
            for bit in code[:-1]:
                if bit == '1':
                    #if the bit is 1 then we need to go the the left child if it exists
                    #or create it first if it does not exist
                    if not current_node.left:
                        current_node.left = Node(None, None)
                    current_node = current_node.left
                else:
                    #else the the bit is 0 then we need to go the the right child if it exists
                    #or create it first if it does not exist
                    if not current_node.right:
                        current_node.right = Node(None, None)
                    current_node = current_node.right
            #inspect the last bit separatley to create a leaf for it. The leaf will contain the symbol
            if code[-1] == '1':
                current_node.left = Leaf(sym, None)
            else:
                current_node.right = Leaf(sym, None)
                
        return root
    
    def _insert_sort(self, node_list):
        '''utility function used by the encode function to sort the node list each iteration'''
        
        for i in range(len(node_list) - 1, 0, -1):
            if node_list[i].probability > node_list[i-1].probability:
                node_list[i], node_list[i-1] = node_list[i-1], node_list[i]
            else:
                return
    
    def _calculate_probabilities(self, sequence):
        '''Given a sequence returns a dictionary with each symbol in the sequence and its probability.'''
        
        #init an empty dict to store the occurence of each character in s
        char_frequencies = {}
        #loop over all characters in s
        for char in sequence:
            #increment the value of the current char by one or create a new key with value 1
            char_frequencies[char] = char_frequencies.get(char, 0) + 1
        #divide the occurence of each character by total number of characters to get the probability
        return {k: v/len(sequence) for k, v in char_frequencies.items()}
        
    
    def _extract_leaves(self, root, symbols, temp_list = []):
        '''utility function to extract the leaves of the huffman tree and use the
        path to each leaf to find the symbol code'''
        
        #if the current node is None then we've reached the end of the branch
        if not root:
            return
        #if the current node has no children then it's a leaf node
        if not root.right and not root.left:
            #store the current bit sequence in the temp_list as the key of this symbol
            symbols[root.symbol] = ''.join([str(bit) for bit in temp_list])
            return
        #add a bit 1 each time we visit a left child
        temp_list.append(1)
        self._extract_leaves(root.left, symbols, temp_list)
        #pop the last bit since we went one level above
        temp_list.pop()
        #add a bit 0 each time we visit a left child
        temp_list.append(0)
        self._extract_leaves(root.right, symbols, temp_list)
        #pop the last bit since we went one level above
        temp_list.pop()
