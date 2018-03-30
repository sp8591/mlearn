# coding:utf-8
import sys, jieba

class WordTocken(object):
    def __init__(self):
        self.START_ID = 4
        self.word2id_dict = {}
        self.id2word_dict = {}

    def load_file_list(self, file_list, min_freq):
        words_count = {}
        for file in file_list:
            with open(file, 'r') as file_object:
                for line in file_object.readlines():
                    line = line.strip()
                    seg_list = jieba.cut(line)
                    for str in seg_list:
                        if str in words_count:
                            words_count[str] = words_count[str] + 1
                        else:
                            words_count[str] = 1
        sorted_list = [[v[1], v[0]] for v in words_count.iteritems()]
        sorted_list.sort(reverse=True)
        for index, item in enumerate(sorted_list):
            word = item[1]
            if item[0] < min_freq:
                break
            self.word2id_dict[word] = self.START_ID + index
            self.id2word_dict[self.START_ID + index] = word
        return index

    def word2id(self, word):
        return self.word2id_dict[word] if word in self.word2id_dict else None

    def id2word(self, id):
        return self.id2word_dict[int(id)] if int(id) in self.id2word_dict else None

