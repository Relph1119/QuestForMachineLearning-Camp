import pickle
import copy
import re
import sys

sys.path.append("..")

# 序列化文件夹
model_path = 'model/hmm.model'
default_probability = 0.000000001
# 转移概率矩阵
trans_mat = {}
# 观测概率矩阵
emit_mat = {}
# 初始概率矩阵
init_vec = {}
# 状态集合
state_set = set()
# 观测集合
observation_set = set()
data_path = 'data/199801人民日报.data'


def train():
    print('begin training......')
    sentences = read_data(data_path)
    for sentence in sentences:
        pre_label = -1
        for word, label in sentence:
            emit_mat[label][word] = emit_mat.setdefault(label, {}).setdefault(word, 0) + 1
            if pre_label == -1:
                init_vec[label] = init_vec.setdefault(label, 0) + 1
            else:
                trans_mat[pre_label][label] = trans_mat.setdefault(pre_label, {}).setdefault(label,0) + 1
            pre_label = label

    for key, value in trans_mat.items():
        number_total = 0
        for k, v in value.items():
            number_total += v
        for k, v in value.items():
            trans_mat[key][k] = 1.0 * v / number_total
    for key, value in emit_mat.items():
        number_total = 0
        for k, v in value.items():
            number_total += v
        for k, v in value.items():
            emit_mat[key][k] = 1.0 * v / number_total

    number_total = sum(init_vec.values())
    for k, v in init_vec.items():
        init_vec[k] = 1.0 * v / number_total

    print('finish training.....')
    save_model()


def predict(text, v_states, start_p, trans_p, obs_p):
    V = [{}]
    path = {}
    # 观测空间
    obs = list(text)
    # 当t=0时进行初始化
    for y in v_states:
        V[0][y] = start_p.get(y, default_probability) * obs_p.get(y, {}).get(obs[0], default_probability)
        path[y] = [y]

    # 当t>0
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for y in v_states:
            (prob, state) = max([(V[t - 1].get(y0, default_probability) *
                                  trans_p.get(y0, {}).get(y, default_probability) *
                                  obs_p.get(y, {}).get(obs[t], default_probability), y0) for y0 in v_states])
            V[t][y] = prob
            new_path[y] = path[state] + [y]
        path = new_path

    (prob, max_state) = max([(V[len(obs) - 1][y], y) for y in v_states])
    result = []
    p = re.compile('BM*E|S')
    for i in p.finditer(''.join(path[max_state])):
        start, end = i.span()
        word = text[start:end]
        result.append(word)
    return result


def load_model():
    print('loading model...')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model[0], model[1], model[2], model[3]


def save_model():
    print('saving model...')
    model = [trans_mat, emit_mat, init_vec, state_set, observation_set]
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def read_data(filename):
    sentences = []
    sentence = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word_label = line.strip().split('\t')
            if len(word_label) == 2:
                observation_set.add(word_label[0])
                state_set.add(word_label[1])
                sentence.append(word_label)
            else:
                # 遇到回车表示一句话结束
                sentences.append(sentence)
                sentence = []
    return sentences



if __name__ == '__main__':
    # 若要重新训练模型，请打开下面的注释
    # train()
    # 加载模型
    trans_mat, emit_mat, init_vec, state_set = load_model()
    # 模型预测
    res = predict("我在图书馆", state_set, init_vec, trans_mat, emit_mat)
    print("//".join(res))
