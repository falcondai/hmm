import numpy as np

def wrap_s_tag(y):
    return ['<s>'] + y + ['</s>']

def estimate_transition(ys):
    bigram_count = {}
    n_bigram = 0
    unigram_count = {}
    n_unigram = 0
    for y in ys:
        context = None
        for t in wrap_s_tag(y):
            if t not in unigram_count:
                unigram_count[t] = 0
            unigram_count[t] += 1
            n_unigram += 1

            if context != None:
                if context not in bigram_count:
                    bigram_count[context] = {}
                if t not in bigram_count[context]:
                    bigram_count[context][t] = 0
                bigram_count[context][t] += 1
                n_bigram += 1
            context = t

    p_transition = {}
    for w1 in bigram_count:
        p_transition[w1] = {}
        for w2 in bigram_count[w1]:
            p_transition[w1][w2] = bigram_count[w1][w2] * 1. / unigram_count[w1]

    return p_transition

def estimate_emission(ys, xs):
    p_emission = {}
    tag_count = {}
    for y, x in zip(ys, xs):
        for tag, token in zip(wrap_s_tag(y), wrap_s_tag(x)):
            if tag not in p_emission:
                p_emission[tag] = {}
            if token not in p_emission[tag]:
                p_emission[tag][token] = 0
            p_emission[tag][token] += 1
            if tag not in tag_count:
                tag_count[tag] = 0
            tag_count[tag] += 1
    for tag in p_emission:
        for token in p_emission[tag]:
            p_emission[tag][token] *= 1. / tag_count[tag]
    return p_emission

def sample_discrete(distribution):
    p = np.cumsum(distribution.values())
    x = np.random.rand()
    for k, pp in zip(distribution.keys(), p):
        if x < pp:
            return k

def sample(p_transition, p_emission):
    y_prev = '<s>'
    x_prev = '<s>'
    y = [y_prev]
    x = [x_prev]
    while y_prev != '</s>':
        y_prev = sample_discrete(p_transition[y_prev])
        y.append(y_prev)
        x_prev = sample_discrete(p_emission[y_prev])
        x.append(x_prev)
    return y, x

def V(m, y, p_transition, p_emission, x, c={}):
    if (m, y) in c:
        return c[m, y]
    if m == 0:
        c[m, y] = 1. if y == '<s>' else 0., [y]
        return c[m, y]

    v_max = 0.
    yhat_max = []
    for yp in p_transition:
        if y in p_transition[yp] and x[m] in p_emission[y]:
            p, yhat = V(m-1, yp, p_transition, p_emission, x, c)
            v = p * p_transition[yp][y] * p_emission[y][x[m]]
            if v_max < v:
                v_max = v
                yhat_max = yhat
    c[m, y] = v_max, yhat_max + [y]
    return v_max, yhat_max + [y]

def inference(x, p_transition, p_emission):
    tagged_x = wrap_s_tag(x)
    m = len(wrap_s_tag(x)) - 1
    return V(m, '</s>', p_transition, p_emission, tagged_x)

def accuracy(y, y_hat):
    n = 0
    for ty, th in zip(y, y_hat[1:-1]):
        if ty == th:
            n += 1
    return n * 1. / len(y_hat)

def evaluate(xs, p_transition, p_emission, ys):
    n = 0
    total_acc = 0.
    for i, (x, y) in enumerate(zip(xs, ys)):
        p, yhat = inference(x, p_transition, p_emission)
        n += 1
        total_acc += accuracy(y, yhat)
    return total_acc / n
