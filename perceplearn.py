import os
import sys


def read_input_files(file_path):
    global truthfulness_class, binary_class
    global all_files_count_dictionary
    global all_vocabulary

    files_counting = []
    for root, directory, files in os.walk(file_path):
        for file in files:
            files_counting.append(root + "/" + file)
            if file == "README.txt":
                continue
            if not file.endswith(".txt"):
                continue

            read_file_path = root + "/" + file

            if 'positive' in read_file_path:
                binary_class[read_file_path] = 1
            elif 'negative' in read_file_path:
                binary_class[read_file_path] = -1

            if 'deceptive' in read_file_path:
                truthfulness_class[read_file_path] = -1
            elif 'truthful' in read_file_path:
                truthfulness_class[read_file_path] = 1

            file_words_list = parse_file(read_file_path)
            all_vocabulary.extend(list(set(file_words_list)))
            counts_dictionary = get_counts_dictionary(file_words_list)
            all_files_count_dictionary[read_file_path] = counts_dictionary


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


def get_counts_dictionary(word_list):
    counts_dict = {}

    for word in word_list:
        if counts_dict.get(word):
            counts_dict[word] += 1
        else:
            counts_dict[word] = 1
    return counts_dict


def vanilla_perceptron():
    global binary_class, truthfulness_class
    global all_files_count_dictionary
    global all_vocabulary
    global binary_weights, truthfulness_weights

    binary_bias = 0.0
    truthfulness_bias = 0.0

    while True:
        converge_binary = True
        converge_truthfulness =  True

        for path in all_files_count_dictionary:
            count_dict = all_files_count_dictionary[path]

            binary_activation = 0
            truthfulness_activation = 0

            for word, count in count_dict.items():
                binary_activation += (binary_weights[word] * count)
                truthfulness_activation += (truthfulness_weights[word] * count)

            binary_activation += binary_bias
            truthfulness_activation += truthfulness_bias

            if (binary_class[path] * binary_activation) <= 0:
                converge_binary = False
                for word, count in count_dict.items():
                    binary_weights[word] += (count * binary_class[path])

                binary_bias += binary_class[path]

            if (truthfulness_class[path] * truthfulness_activation) <= 0:
                converge_truthfulness = False
                for word, count in count_dict.items():
                    truthfulness_weights[word] += (count * truthfulness_class[path])

                truthfulness_bias += truthfulness_class[path]

        if converge_truthfulness and converge_binary:
            break

    return [binary_bias, truthfulness_bias, binary_weights, truthfulness_weights]


def average_perceptron():
    global all_files_count_dictionary
    global binary_class, truthfulness_class
    global all_vocabulary
    global binary_avg_weights, truthfulness_avg_weights, cached_binary_weights, cached_truthfulness_weights

    binary_bias = 0
    truthfulness_bias = 0
    binary_beta = 0
    truthfulness_beta = 0

    counter = 1

    while True:
        converge_binary = True
        converge_truthfulness = True

        for path in all_files_count_dictionary:
            count_dict = all_files_count_dictionary[path]

            binary_activation = 0
            truthfulness_activation = 0

            for word, count in count_dict.items():
                binary_activation += (binary_avg_weights[word] * count)
                truthfulness_activation += (truthfulness_avg_weights[word] * count)

            binary_activation += binary_bias
            truthfulness_activation += truthfulness_bias

            if (binary_class[path] * binary_activation) <= 0:
                converge_binary = False
                for word, count in count_dict.items():
                    binary_avg_weights[word] += (count * binary_class[path])
                    cached_binary_weights[word] += (counter * count * binary_class[path])

                binary_bias += binary_class[path]
                binary_beta += (counter * binary_class[path])

            if (truthfulness_class[path] * truthfulness_activation) <= 0:
                converge_truthfulness = False
                for word, count in count_dict.items():
                    truthfulness_avg_weights[word] += (count * truthfulness_class[path])
                    cached_truthfulness_weights[word] += (count * counter * truthfulness_class[path])

                truthfulness_bias += truthfulness_class[path]
                truthfulness_beta += (counter * truthfulness_class[path])

            counter += 1

        if converge_binary and converge_truthfulness:
            break

    for word in binary_avg_weights:
        binary_avg_weights[word] -= float(cached_binary_weights[word]) /counter

    for word in truthfulness_avg_weights:
        truthfulness_avg_weights[word] -= float(cached_truthfulness_weights[word]) / counter

    binary_bias -= float(binary_beta) / counter
    truthfulness_bias -= float(truthfulness_beta) / counter

    return [binary_bias, truthfulness_bias, binary_avg_weights, truthfulness_avg_weights]


def write_output_file(name, results):
    global all_vocabulary

    filename = name + 'model.txt'

    with open(filename, 'w') as out_file:
        out_file.write('binary_bias' + " | " + str(results[0]) + "\n")
        out_file.write('truthfulness_bias' + " | " + str(results[1]) + "\n")

        for word in all_vocabulary:
            out_file.write(str(word) + " | " + str(results[2][word]) +"\n")
            out_file.write(str(word) + " | " + str(results[3][word]) +"\n")


if __name__== "__main__":
    input_file_path = sys.argv[1]

    all_vocabulary = []
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

    binary_class = dict()
    truthfulness_class = dict()

    read_input_files(input_file_path)

    binary_weights = dict()
    truthfulness_weights = dict()
    binary_avg_weights = dict()
    truthfulness_avg_weights = dict()
    cached_binary_weights = dict()
    cached_truthfulness_weights = dict()

    all_vocabulary = set(all_vocabulary)

    print len(all_vocabulary)

    for word in all_vocabulary:
        binary_weights[word] = 0
        truthfulness_weights[word] = 0
        binary_avg_weights[word] = 0
        truthfulness_avg_weights[word] = 0
        cached_binary_weights[word] = 0
        cached_truthfulness_weights[word] = 0


    vanilla_results = vanilla_perceptron()
    average_results = average_perceptron()

    write_output_file('vanilla', vanilla_results)
    write_output_file('average', average_results)

