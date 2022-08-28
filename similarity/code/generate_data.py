import ccy
import torch

import stanza
nlp = stanza.Pipeline(lang = 'en', processors = 'tokenize, pos, constituency', tokenize_pretokenized=True)

#import translators as ts

from sentence_transformers import SentenceTransformer, util

import docx
from docx import Document



def get_sentences_from_24301():

    lst_sentence = []

    dic_para = ccy.get_paragraph()
    for para_ID in dic_para:
        if dic_para[para_ID]['style'].startswith('H'):
            continue
        para_text = ccy._3gpp_get_para_text_without_format(dic_para[para_ID]['text'])
        para_sents = ccy.get_spacy_sents_from_a_para(para_text)
        lst_sentence += para_sents

    return lst_sentence


def get_sentences_from_3gpp_spec():

    folder = '/home/tangd/chen481/ConfTest/spec_info/data/3gpp_spec/'
    filepath_list = ['%s/24301-g80.docx'%folder,
                     '%s/24008-f90.docx'%folder,
                     '%s/24501-f60.docx'%folder,
                     '%s/33401-fb0.docx'%folder,
                     '%s/33501-fd0.docx'%folder,
                     '%s/36331-fe0.docx'%folder,
                     ]

    lst_sentence = []

    for filepath in filepath_list:

        document = Document(filepath)

        for para in document.paragraphs:
            para_text = para.text.strip()
            if len(para_text) > 0:
                para_text = ccy._3gpp_get_para_text_without_format(para_text).strip()
                if len(para_text) > 0:
                    para_sents = ccy.get_spacy_sents_from_a_para(para_text)
                    lst_sentence += para_sents

    return lst_sentence


def get_partial_sentences_based_on_constituency(node):

    lst = []

    if node.label == 'S' and len(node.children) > 0:
        children_label = [child.label for child in node.children]
        if 'NP' in children_label and 'VP' in children_label:
            lst.append(get_text_on_constituency_tree(node, []))

    for child in node.children:
        lst += get_partial_sentences_based_on_constituency(child)

    return lst


def generate_pre_sentences():

    lst_sentence = get_sentences_from_24301()

    #lst_sentence = get_sentences_from_3gpp_spec()

    file = open('../data/pre_sentences.txt', 'w')

    cnt = 0
    cnt_generate = 0

    for sentence in lst_sentence:

        try:
            doc = nlp([sentence.split(' ')])
            root = doc.sentences[0].constituency

            partial_sentences = get_partial_sentences_based_on_constituency(root)

            for ps in partial_sentences:
                if ps in lst_sentence:
                    if len(ps.strip().split(' ')) <= 128:
                        file.write('%s\n'%ps.strip())
                        cnt_generate += 1
        except:
            continue

        cnt += 1
        if cnt % 500 == 0:
            print(cnt, '/', len(lst_sentence))
            print('generate:', cnt_generate)
            ccy.print_time()

    file.close()



def get_sentences_from_prepared():

    lst = []

    file = open('../data/pre_sentences.txt', 'r')
    fileContent = file.readlines()
    file.close()

    for tmp in fileContent:
        if len(tmp.strip().split(' ')) <= 128:
            lst.append(tmp.strip())

    return lst


#############
#           #
#  label 0  #
#           #
#############

def generate_data_for_label_0():

    lst_sentence = get_sentences_from_prepared()

    file = open('../data/training_set_for_label_0', 'w')

    cnt = 0
    cnt_generate = 0

    for sent_1 in lst_sentence:

        verb_list = [' shall ', ' should ', ' must ', ' will ', ' may ', ' is ', ' are ', ' were ', ' do ', ' does ']
        for verb in verb_list:
            sent_2 = sent_1.replace(verb, '%snot '%verb)
            while ' not not ' in sent_2:
                sent_2 = sent_2.replace(' not not ', ' not ')
            while '  ' in sent_2:
                sent_2 = sent_2.replace('  ', ' ')
            if sent_1 != sent_2:
                file.write('0\n%s\n%s\n'%(sent_1, sent_2))
                file.write('0\n%s\n%s\n'%(sent_2, sent_1))
                cnt_generate += 1

            sent_2 = sent_1.replace('%snot'%verb, verb)
            while '  ' in sent_2:
                sent_2 = sent_2.replace('  ', ' ')
            if sent_1 != sent_2:
                file.write('0\n%s\n%s\n'%(sent_1, sent_2))
                file.write('0\n%s\n%s\n'%(sent_2, sent_1))

        replace_pair = [(' with ', ' without '), (' has been ', ' has not been '), (' have been ', ' have not been ')]
        for (a, b) in replace_pair:
            sent_2 = sent_1.replace(a, b)
            while '  ' in sent_2:
                sent_2 = sent_2.replace('  ', ' ')
            if sent_1 != sent_2:
                file.write('0\n%s\n%s\n'%(sent_1, sent_2))
                file.write('0\n%s\n%s\n'%(sent_2, sent_1))
                cnt_generate += 1

            sent_2 = sent_1.replace(b, a)
            while '  ' in sent_2:
                sent_2 = sent_2.replace('  ', ' ')
            if sent_1 != sent_2:
                file.write('0\n%s\n%s\n'%(sent_1, sent_2))
                file.write('0\n%s\n%s\n'%(sent_2, sent_1))
                cnt_generate += 1

        cnt += 1
        if cnt%1000 == 0:
            print(cnt, '/', len(lst_sentence))
            print('generate:', cnt_generate)
            ccy.print_time()

    file.close()



###########
#         #
# label 1 #
#         #
###########

def generate_data_for_label_1_using_translation():

    lst_sentence = get_sentences_from_prepared()

    file = open('../data/training_set_for_label_1.translation', 'w')

    cnt = 0

    for i in range(2598, len(lst_sentence)):

        sent_1 = lst_sentence[i]

        sent_de = ts.google(sent_1, from_language = 'en', to_language = 'de')
        sent_2 = ts.google(sent_de, from_language = 'de', to_language = 'en')
        sent_zh = ts.google(sent_1, from_language = 'en', to_language = 'zh')
        sent_3 = ts.google(sent_zh, from_language = 'zh', to_language = 'en')

        file.write('1\n%s\n%s\n'%(sent_1, sent_2))
        file.write('1\n%s\n%s\n'%(sent_1, sent_3))
        file.write('1\n%s\n%s\n'%(sent_2, sent_3))

        cnt = cnt + 1
        print('%d/%d:'%(cnt, len(lst_sentence)))
        ccy.print_time()

        if cnt % 100 == 0:
            file = file.close()
            file = open('../data/training_set_for_label_1.v1.translation', 'a')

    file.close()


def generate_data_for_label_1_using_cosine():

    lst_sentence = get_sentences_from_prepared()

    model = SentenceTransformer('all-mpnet-base-v2')

    lst_sentence_embedding = model.encode(lst_sentence, convert_to_tensor=True, normalize_embeddings=True)

    print(lst_sentence_embedding.shape)

    a = torch.matmul(lst_sentence_embedding, torch.transpose(lst_sentence_embedding,0,1))
    a = a.detach().cpu().numpy()

    file = open('../data/training_set_for_label_1.cosine', 'w')

    for i in range(len(lst_sentence)):
        for j in range(len(lst_sentence)):
            if lst_sentence[i] == lst_sentence[j]:
                continue
            if a[i][j] > 0.9:
                file.write('1\n%s\n%s\n'%(lst_sentence[i], lst_sentence[j]))

    file.close()


def generate_data_for_label_1_using_t1():

    lst_sentence = get_sentences_from_prepared()

    file = open('../data/training_set_for_label_1.t1', 'w')

    cnt = 0
    cnt_generate = 0

    for sent_1 in lst_sentence:

        pair_2_to_1 = [(' GUTI reallocation procedure', ' GUTI reallocation '),
                       (' authentication procedure', ' authentication '),
                       (' security mode control procedure', ' security mode control '),
                       (' identification procedure', ' identification '),
                       (' attach procedure', ' attach '),
                       (' detach procedure', ' detach '),
                       (' tracking area updating procedure', ' tracking area updating '),
                       (' service request procedure', ' service request '),
                       (' paging procedure', ' paging ')
                         ]

        for (p1, p2) in pair_2_to_1:
            if p1 in sent_1:
                sent_2 = sent_1.replace(p1, p2)
                while '  ' in sent_2:
                    sent_2 = sent_2.replace('  ', ' ')
                if sent_2.endswith(' .'):
                    sent_2 = sent_2[:-2] + '.'

                file.write('1\n%s\n%s\n'%(sent_1, sent_2))
                file.write('1\n%s\n%s\n'%(sent_2, sent_1))

                cnt_generate += 2

        cnt += 1
        if cnt % 10000 == 0:
            print(cnt, '/', len(lst_sentence))
            print('generate:', cnt_generate)
            ccy.print_time()

    file.close()


def generate_data_for_label_1_using_t2():

    lst_sentence = get_sentences_from_prepared()

    file = open('../data/training_set_for_label_1.t2', 'w')

    cnt = 0
    cnt_generate = 0

    for sent_1 in lst_sentence:

        try:
            doc = nlp([sent_1.split(' ')])
            root = doc.sentences[0].constituency

            if len(root.children) == 1 and root.children[0].label == 'S':
                node = root.children[0]
                NP_text = None

                if len(node.children) == 2:
                    child_1 = node.children[0]
                    child_2 = node.children[1]
                    if child_1.label == 'NP' and child_2.label == 'VP':
                        NP_text = get_text_on_constituency_tree(child_1, [])

                if NP_text != None:
                    if NP_text in ['UE', 'the UE', 'The UE',
                                    'MME', 'the MME', 'The MME',
                                    'network', 'the network', 'The network']:
                        sent_2 = get_text_on_constituency_tree(child_2, [])

                        file.write('1\n%s\n%s\n'%(sent_1, sent_2))
                        file.write('1\n%s\n%s\n'%(sent_2, sent_1))

                        cnt_generate += 1
        except:
            continue

        cnt += 1
        if cnt%100 == 0:
            print(cnt, '/', len(lst_sentence))
            print('generated:', cnt_generate)
            ccy.print_time()

    file.close()



def generate_data_for_label_1():

    #generate_data_for_label_1_using_translation()

    generate_data_for_label_1_using_cosine()

    # XXX procedure = procedure
    #generate_data_for_label_1_using_t1()

    # remove subjective
    #generate_data_for_label_1_using_t2()


###################
#                 #
#  label 2 and 3  #
#                 #
###################


def get_text_on_constituency_tree_dfs(node, to_delete_node):

    if node in to_delete_node:
        return ''

    if len(node.children) == 0:
        return node.label

    text = ''
    for child in node.children:
        text = text + ' ' + get_text_on_constituency_tree_dfs(child, to_delete_node)

    return text


def get_text_on_constituency_tree(node, to_delete_node):

    text = get_text_on_constituency_tree_dfs(node, to_delete_node)

    while '  ' in text:
        text = text.replace('  ', ' ')

    if text.endswith(' .'):
        text = text[:-2] + '.'

    text = text.strip()

    return text


def s1(root, node, label_layers, ancestors):

    lst = []

    if len(node.children) > 0:

        children_labels = [child.label for child in node.children]

        if node.label == 'NP' and 'JJ' in children_labels:
            if ancestors[-1] == 'S':
                to_delete_node = []
                for child in node.children:
                    if child.label == 'JJ':
                        to_delete_node.append(child)
                lst.append(get_text_on_constituency_tree(root, to_delete_node))

        if node.label == 'NP' and len(node.children) == 2 and children_labels[0] == 'NP' and children_labels[1] == 'VP':
            if (len(ancestors) >= 3 and ancestors[-1] == 'VP' and ancestors[-2] == 'S' and ancestors[-3] == 'ROOT') or \
               (len(ancestors) >= 4 and ancestors[-1] == 'VP' and ancestors[-2] == 'VP' and ancestors[-3] == 'S' and ancestors[-4] == 'ROOT'):
                to_delete_node = [node.children[1]]
                lst.append(get_text_on_constituency_tree(root, to_delete_node))

        label_layers.append(children_labels)
        ancestors.append(node.label)
        for child in node.children:
            lst += s1(root, child, label_layers, ancestors)
        label_layers.pop()
        ancestors.pop()

    return lst



def generate_data_for_label_s():

    lst_sentence = get_sentences_from_prepared()

    f2 = open('../data/training_set_for_label_2', 'w')
    f3 = open('../data/training_set_for_label_3', 'w')

    cnt = 0
    cnt_generate = 0

    for sent_1 in lst_sentence:

        try:
            doc = nlp([sent_1.split(' ')])
            root = doc.sentences[0].constituency

            lst_sent_2 = s1(root, root, [], [])

            for sent_2 in lst_sent_2:

                f2.write('2\n%s\n%s\n'%(sent_1, sent_2))
                f3.write('3\n%s\n%s\n'%(sent_2, sent_1))

                cnt_generate += 1
        except:
            continue

        cnt += 1
        if cnt % 100 == 0:
            print(cnt, '/', len(lst_sentence))
            print('generated:', cnt_generate)
            ccy.print_time()

    f2.close()
    f3.close()




if __name__ == '__main__':

    generate_pre_sentences()

    #generate_data_for_label_0()

    #generate_data_for_label_1()

    #generate_data_for_label_s()
