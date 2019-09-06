import sys
import os
from itertools import islice


def read_model_file(file_name):
    global binary_bias, truthfulness_bias
    global binary_weights, truthfulness_weights

    with open(file_name, 'r') as model_file:
        for n_lines in iter(lambda: tuple(islice(model_file, 2)), ()):
            l1 = n_lines[0].split(" ")
            l2 = n_lines[1].split(" ")

            if l1[0] == 'binary_bias':
                binary_bias = float(l1[-1])
            elif l2[0] == 'truthfulness_bias':
                truthfulness_bias = float(l2[-1])

            binary_weights[l1[0]] = float(l1[-1])
            truthfulness_weights[l2[0]] = float(l2[-1])


def run_perceptron(file_path):
    global binary_bias, truthfulness_bias
    global binary_weights, truthfulness_weights
    global all_files_count_dictionary

    files_counting = []
    for root, directory, files in os.walk(file_path):
        for file in files:
            files_counting.append(root + "/" + file)
            if file == "README.txt":
                continue
            if not file.endswith(".txt"):
                continue

            read_file_path = root + "/" + file

            file_words_list = parse_file(read_file_path)
            counts_dictionary = get_counts_dictionary(file_words_list)
            all_files_count_dictionary[read_file_path] = counts_dictionary

    with open('percepoutput.txt', 'w') as out_file:

        for path in all_files_count_dictionary:
            count_dict = all_files_count_dictionary[path]
            binary_value = 0
            truthfulness_value = 0
            for word, count in count_dict.items():
                if word in binary_weights:
                    binary_value += binary_weights[word] * count
                if word in truthfulness_weights:
                    truthfulness_value += truthfulness_weights[word] * count

            binary_value += binary_bias
            truthfulness_value += truthfulness_bias

            label1, label2 = '', ''

            if truthfulness_value > 0:
                label1 = 'truthful'
            else:
                label1 = 'deceptive'

            if binary_value > 0:
                label2 = 'positive'
            else:
                label2 = 'negative'

            out_file.write(label1 + " " + label2 + " " + path + "\n")


def get_counts_dictionary(word_list):
    counts_dict = {}

    for word in word_list:
        if counts_dict.get(word):
            counts_dict[word] += 1
        else:
            counts_dict[word] = 1
    return counts_dict


def parse_file(file_path):
    global stopwords_list

    with open(file_path, 'r') as input_file:
        for line in input_file:
            # Remove punctuation
            line = line.replace(".", " ")
            line = line.replace("-", " ")
            # Remove empty lines and empty spaces
            words = line.split()

            word_list = []
            for word in words:
                word = word.strip('`-~ _,;]{}[!?.:<>(*)@#$\'\"+=&^%1234567890\n')
                word = word.lower()
                if word not in stopwords_list and word != "":
                    word_list.append(word)

    return word_list


if __name__== "__main__":
    input_file_name = sys.argv[1]
    input_file_path = sys.argv[2]

    all_files_count_dictionary = dict()

    stopwords_list = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about",
                      "once", "during", "out", "very", "having", "with", "they", "own", "an", "be",
                      "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself",
                      "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each",
                      "the", "themselves", "until", "below", "are", "we", "these", "your", "his",
                      "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down",
                      "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had",
                      "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been",
                      "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what",
                      "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself",
                      "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after",
                      "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing",
                      "it", "how", "further", "was", "here", "than"]

    binary_bias = 0
    truthfulness_bias = 0
    binary_weights = dict()
    truthfulness_weights = dict()

    read_model_file(input_file_name)

    run_perceptron(input_file_path)