#!/usr/bin/env python
#coding=utf-8
from __future__ import print_function
#from tensorflow.python import debug as tf_debug
import tensorflow as tf
from sklearn.decomposition import PCA
import sys
import random
import numpy as np
import os
import math

Pctr_oneday = 0.17579
subcat = 23
collect = 3
components_num = 40
Bias = 15
WIDTH = subcat * collect
GAME = 'recommendation' # the name of the game being played for log files
ACTIONS = 23 # number of valid actions
GAMMA = 0.0# decay rate of past observations
TAU = 0.001
BATCH = 1 # size of minibatch
TwoRow = True
#FRAME_PER_ACTION = 1
DUELING = True
data_prefix =  './data/include_pctr/part-00099'
f = open(data_prefix, 'r')
D = f.readlines()
f.close()
r_scale = 0.5
click_bias = 6
#r_b = r_scale * (Pctr_oneday * click_bias + (1- Pctr_oneday) * -1)
r_b = r_scale * -1
potential_scale = 0.0


f = open('./data/potential', 'r')
potential = f.read().strip('|').split('|')
f.close()

DEBUG = False
Use_PCA= False
#Use_Bias = True
#if debug, it will retrieve the data in the sequence order

'''
W1 = tf.Variable(0.01*np.array(range(WIDTH*128)).reshape(WIDTH,128),dtype=tf.float32)
W2 = tf.Variable(0.01*np.array(range(23*128)).reshape(128,ACTIONS),dtype=tf.float32)
B1 = tf.Variable(0.01*np.array(range(128)),dtype=tf.float32)
B2 = tf.Variable(0.01*np.array(range(ACTIONS)),dtype=tf.float32)
'''
educate = '高中及以下|大专|本科及以上|'
sex = '男|女|'
age = '35-44|18-24|25-34|45-54|18以下|55-64|65以上|error'
content = '汽车|娱乐|生活|影视|时尚|科技|游戏|社会|军事|动漫|亲子|搞笑趣味|财经|美食|体育|文化历史|大自然|音乐|其他|舞蹈|旅游|教育|时政|'
#bias + state


age_list = age.rstrip('|').split('|')
age_dict = {age:i for i,age in enumerate(age_list)}
age_dict[""] = 7

sex_list = sex.rstrip('|').split('|')
sex_dict = {sex:i for i,sex in enumerate(sex_list)}
sex_dict[""] = 2

educate_list = educate.rstrip('|').split('|')
educate_dict = {educate:i for i,educate in enumerate(educate_list)}
educate_dict[""] = 3

cate_list = content.rstrip('|').split('|')
cate_dict = {cate:i for i,cate in enumerate(cate_list)}
#cate_dict[""] = 20

cate_reverse_dict = {i:cate for i,cate in enumerate(cate_list)}


a_m = len(age_list)
s_m = len(sex_list)
e_m = len(educate_list)
c_m = len(cate_list)
j = [a_m, s_m, e_m, c_m, c_m, c_m]
k = [reduce(lambda x, y: x * y, j[i + 1:]) if i != len(j) - 1 else 1 for i in range(len(j))]

def potential_part(bias_str,state_cates,next_state_cates):
    try:
        a, s, e = bias_str.split('@')
    except ValueError:
        return 0
    C_this = state_cates.rstrip('@\n').split('@')
    C_next = next_state_cates.rstrip('@\n').split('@')
    t_this = [age_dict[a],sex_dict[s],educate_dict[e],cate_dict[C_this[0]],cate_dict[C_this[1]],cate_dict[C_this[2]]]
    t_next = [age_dict[a],sex_dict[s],educate_dict[e],cate_dict[C_next[0]],cate_dict[C_next[1]],cate_dict[C_next[2]]]
    index_this = sum([t_this[i] * k[i] for i in range(len(j))])
    index_next = sum([t_next[i] * k[i] for i in range(len(j))])
    return potential_scale*(GAMMA* float(potential[index_next]) - float(potential[index_this]))



'''
tag_vector_text = open('tag_vector').readlines()
tag_vector_split = [each.rstrip(' \n').split(' ') for each in tag_vector_text]
tag_vector = {each[0]: map(float,each[1:]) for each in tag_vector_split}
tag_vector[''] = [0.0] * 100


'''
def weight_variable(shape,scope,trainable,scale):
    with tf.variable_scope(scope):
        initial = scale*tf.truncated_normal(shape = shape)
        return tf.Variable(initial,trainable=trainable)

def bias_variable(shape,scope,trainable,scale):
    with tf.variable_scope(scope):
        initial = scale*tf.random_normal(shape = shape)
        return tf.Variable(initial,trainable=trainable)

Q_max = []
Loss = []

Q_file = 'Q_max'
Loss_file = 'Loss'
Recom_file = 'recom'
Auc_file = 'Auc'

global_step = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False)



class Network(object):
    def __init__(self,sess,Dueling):
        self.Dueling = Dueling

        with tf.variable_scope('critical'):
            self.v_main,self.q_main,self.variable_list_main,self.b_main = \
                self.createDeepNet('Main_Net',True,scale=0.0)
            self.v_target, self.q_target, self.variable_list_target,self.b_target =\
                self.createDeepNet('Target_Net',False,scale=0.0)
            self.assign_list = [self.variable_list_target[i].assign(tf.multiply(TAU , self.variable_list_main[i]) +
                                               tf.multiply((1.0 - TAU) , self.variable_list_target[i]))
                                            for i in range(len(self.variable_list_main))]

        with tf.variable_scope('update'):
            self.sess = sess
            self.a = tf.placeholder(dtype=tf.float32, shape=[None, ACTIONS], name='a')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')
            self.q_action_train = tf.reduce_sum(tf.multiply(self.q_main, self.a))
            self.Loss = tf.reduce_mean(tf.square(self.y - self.q_action_train))
            #self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.Loss,var_list=self.variable_list_main,
                                                                     #global_step=global_step)
            self.gradient = tf.gradients(ys=self.Loss, xs=self.variable_list_main)
            self.optimize = tf.train.AdadeltaOptimizer()
            self.train_step = self.optimize.apply_gradients(zip(self.gradient, self.variable_list_main), global_step=global_step)
    def createDeepNet(self,scope,trainable,scale):
        with tf.variable_scope(scope):
        # input layer
            v = tf.placeholder(dtype=tf.float32, shape=[None,1,ACTIONS*collect],name='v')
            b = tf.placeholder(dtype=tf.float32, shape=[None,1,Bias],name='b')
            v_flatten = tf.reshape(v, [-1, collect*ACTIONS])
            b_flateen = tf.reshape(b, [-1, Bias])
            s = tf.concat([v_flatten,b_flateen],axis=1)


            #w1_b = weight_variable([Bias, ACTIONS], scope, trainable, scale)
            w1_a = weight_variable([Bias + WIDTH, ACTIONS], scope, trainable, scale)
            w1_v = weight_variable([Bias + WIDTH, 1],scope,trainable,scale)

            A = tf.matmul(s, w1_a)
            V = tf.matmul(s, w1_v)

            q = V + A - tf.reduce_mean(A, reduction_indices=1, keep_dims=True)


            #cv1_w = weight_variable([1, 4, collect, 16], scope, trainable, scale)
            #cv1_b = bias_variable([16],scope,trainable,scale)
            #h_conv1 = tf.nn.conv2d(v, cv1_w, strides=[1, 1, 2, 1], padding="SAME") +cv1_b
            #h_conv1_flateen = tf.reshape(h_conv1, [-1, 192])

            #fc1_w = weight_variable([192, ACTIONS],scope,trainable,scale)
            #fc1_b = bias_variable([ACTIONS],scope,trainable,scale)

            #fc2_b = bias_variable([ACTIONS], scope, trainable, scale)

            #s_expand = tf.expand_dims(s,axis=3)
            #fc1_h = tf.matmul(h_conv1_flateen, fc1_w) + fc1_b


            #fc2_w = weight_variable([256, ACTIONS], scope, trainable, scale)
            #fc2_b = bias_variable([ACTIONS],scope,trainable,scale)

            variable_list = [w1_a,w1_v]

            '''
            # hidden layers
            cv1_w = weight_variable([1,10,1,32],scope,trainable,scale)
            cv1_b = bias_variable([32],scope,trainable,scale)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(value=tf.nn.conv2d(s_expand, cv1_w, strides = [1,1,5,1], padding = "VALID"),
                                                bias=cv1_b,data_format='NHWC'))

            #h1_max_pooling = tf.nn.max_pool(h_conv1,ksize=[1,1,3,1],strides=[1,1,3,1],padding='SAME')


            cv2_w = weight_variable([1,5,32,64],scope,trainable,scale)
            cv2_b = bias_variable([64],scope,trainable,scale)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(value=tf.nn.conv2d(h_conv1, cv2_w, strides=[1,1,2,1], padding="SAME"),
                                                bias=cv2_b, data_format='NHWC'))

            #h2_max_pooling = tf.nn.max_pool(h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

            h_conv2_flat = tf.reshape(h_conv2, [-1, 3264])

            fc2_w = weight_variable([3264,400],scope,trainable,scale)
            fc2_b = bias_variable([400],scope,trainable,scale)
            h2 = tf.nn.relu(tf.matmul(h_conv2_flat, fc2_w) + fc2_b)

            if self.Dueling:
                #A
                fc3_w_a = weight_variable([400, ACTIONS], scope, trainable,scale)
                fc3_b_a = bias_variable([ACTIONS], scope, trainable,scale)
                A = (tf.matmul(h2, fc3_w_a) + fc3_b_a)
                #V
                fc3_w_v = weight_variable([400, 1], scope, trainable,scale)
                fc3_b_v = bias_variable([1], scope, trainable,scale)
                V = (tf.matmul(h2, fc3_w_v) + fc3_b_v)
                q = V + A - tf.reduce_mean(A, reduction_indices=1, keep_dims=True)
                variable_list = [cv1_w,cv1_b,cv2_w,cv2_b,fc2_w,fc2_b,fc3_w_a,fc3_b_a,fc3_w_v,fc3_b_v]

            else:
                fc3_w = weight_variable([400, ACTIONS],scope,trainable,scale)
                fc3_b = bias_variable([ACTIONS],scope,trainable,scale)
                q = (tf.matmul(h2, fc3_w) + fc3_b)

                variable_list = [cv1_w,cv1_b,cv2_w,cv2_b,fc2_w,fc2_b,fc3_w,fc3_b]
            '''

        return v,q,variable_list,b


        #hidden_variable = [h_cv1,h_cv2]
        #def train_q_net(self,target,a):
    def train_main(self,a,y,v,b):
        _,Loss = self.sess.run([self.train_step,self.Loss],{
            self.a: a,
            self.y: y,
            self.v_main : v,
            self.b_main: b
        })
        return Loss
    def target_predict(self,v,b):
        return self.sess.run(self.q_target,{
            self.v_target:v,
            self.b_target:b
        })

    def main_predict(self,v,b):
        return self.sess.run(self.q_main,{
            self.v_main:v,
            self.b_main:b
        })

    def train_target(self):
        self.sess.run(self.assign_list)



#index[num * subcat + cate_dict[val.rstrip('\n')]] = 1
def one_hot_state(array_1d):
    index = np.zeros([subcat * collect])
    for num,val in enumerate(array_1d):
        if val != "":
            if num == 0:
                index[num * subcat + cate_dict[val.rstrip('\n')]] = 6
            else:
                index[num * subcat + cate_dict[val.rstrip('\n')]] = 1
    return index


#def pca_s_one_hot(array_1d):
    #return  np.transpose(pca.transform(np.expand_dims(one_hot_state(array_1d),axis=0)))

def one_hot_action(value):
    index = np.zeros([len(cate_dict.keys()),])
    index[value] = 1
    return index

def one_hot_bias(one_user):
    age, sex, educate = one_user.split('@')
    age_one_hot = np.zeros([8,])
    if age != '':
        age_one_hot[age_dict[age]] = 1
    sex_one_hot = np.zeros([len(sex_dict.keys()),])
    if sex != '':
        sex_one_hot[sex_dict[sex]] = 1
    educate_one_hot = np.zeros([len(educate_dict.keys()),])
    if educate != '':
        educate_one_hot[educate_dict[educate]] = 1
    return np.append(np.append(age_one_hot,sex_one_hot),educate_one_hot)

'''
def transfer_tag(tags):
    list_tag_vector = reduce(lambda x, y: x + y, [np.array(tag_vector[each]) for each in tags.split('#')]) / (
        1.0 * len(tags.split('#')))
    return list_tag_vector

def transfer_state(readlist):
    readlist_split = readlist.rstrip('\t\n').split('|')
    read_component = map(transfer_tag,readlist_split)
    state_vector = reduce(lambda x,y:np.append(x,y),read_component)
    return state_vector

'''


def extract_data(size,t):
    if DEBUG:
        minibatch = [D[t]]
    else:
        minibatch = random.sample(D, size)
    #print('\t'.join(minibatch))
    minibatch_split = [x.split('\t') for x in minibatch]
    Bias = [d[1] for d in minibatch_split]
    rest = [d[2] for d in minibatch_split]
    #Bias_one_hot = map(one_hot_bias, Bias)
    rest_split = [x.rsplit('\t\n ')[0].split('$') for x in rest]
    Bias_one_hot=[]
    v_j_batch =[]
    v_j1_batch =[]
    r_batch =[]
    a_batch =[]
    a_batch_one_hot =[]
    click_batch = []
    for i in range(len(Bias)):
        try:
            vj = one_hot_state(rest_split[i][0].split('@'))
        except ValueError:
            continue
        try:
            vj1 = one_hot_state(rest_split[i][3].split('@'))
        except ValueError:
            continue
        try:
            a = cate_dict[rest_split[i][2]]
            a_hot = one_hot_action(a)
        except ValueError:
            continue

        #p = potential_next(Bias[i], rest_split[i][3])
        p = potential_part(Bias[i], rest_split[i][0], rest_split[i][3])
        global r_b
        if int(rest_split[i][1].split('@')[0]) == 1:
            #r = click_bias* r_scale*float(rest_split[i][1].split('@')[1]) - r_b
            r = click_bias * r_scale - r_b
        else:
            #r = -r_scale*float(rest_split[i][1].split('@')[1]) - r_b
            r = -r_scale  - r_b
        #r = r_scale*float(rest_split[i][1].split('@')[1] if rest_split[i][1].split('@')[0] == '1' else -rest_split[i][1].split('@')[1])
        r += p
        try:
            b_one_hot = one_hot_bias(Bias[i])
        except ValueError:
            continue
        Bias_one_hot.append(b_one_hot)
        v_j_batch.append(vj)
        v_j1_batch.append(vj1)
        a_batch.append(a)
        a_batch_one_hot.append(a_hot)
        r_batch.append(r)
        click_batch.append(int(rest_split[i][1].split('@')[0]))

    #s_j_tags_batch = [d[1] for d in minibatch_split]
    #s_j_vector = map(transfer_state, s_j_tags_batch)
    #s_j_batch = [map(float,r[0].split('|')) for r in rest_split]
    #s_j1_batch = [map(float,r[3].split('|')) for r in rest_split]
    #s_j_batch = [map(one_hot_state,r[0].split('|')) for r in rest_split]
    #s_j1_batch = [map(one_hot_state,r[3].split('|')) for r in rest_split]
    #r_batch = [100 if int(r[1]) == 1 else -10 for r in rest_split]
    #a_batch = [cate_dict[r[2]] for r in rest_split]
    #a_hot = map(one_hot_action,a_batch)
    #s_j1_tags_batch = [d[4] for d in minibatch_split]
    #s_j1_vector = map(transfer_state, s_j1_tags_batch)

    #print('s_j_batch',np.array(s_j_batch).shape)
    #print('Bias_one_hot', np.array(Bias_one_hot).shape)
    #print('s_j1_batch', np.array(s_j1_batch).shape)

    #s_j_batch_hot_bias = np.expand_dims(np.append(np.array(Bias_one_hot), np.array(s_j_batch),axis=1),axis=1)
    #s_j1_batch_hot_bias = np.expand_dims(np.append(np.array(Bias_one_hot), np.array(s_j1_batch),axis=1),axis=1)

    v_j_batch_expend = np.expand_dims(np.array(v_j_batch),axis=1)
    v_j1_batch_expend = np.expand_dims(np.array(v_j1_batch), axis=1)
    Bias_one_hot_expend = np.expand_dims(np.array(Bias_one_hot), axis=1)

    return v_j_batch_expend, v_j1_batch_expend,Bias_one_hot_expend,r_batch,click_batch, a_batch_one_hot, a_batch


if Use_PCA:
    PCA_data_num = 1000000
    s_batch,_,_,_,_ = extract_data(PCA_data_num)
    pca = PCA(n_components=components_num)
    pca.fit(s_batch)


def transfer_data(size,t):
    if Use_PCA:
        assert TwoRow != True,'PCA can not be used in TwoRow Mode'
        s_j_batch_hot_bias, r_batch, a_hot, s_j1_batch_hot_bias, a_batch =  extract_data(size)
        s_j_transfer = pca.transform(s_j_batch_hot_bias)
        s_j1_transfer = pca.transform(s_j1_batch_hot_bias)
        return s_j_transfer,r_batch,a_hot,s_j1_transfer,a_batch
    else:
        return extract_data(size,t)



def simulate_recom(DQN,user1_like,user1_dislike,user1_bias):
    show_tag = '@'.join([user1_like,user1_dislike])
    b_hot = one_hot_bias(user1_bias)
    v_j_one_hot = one_hot_state(show_tag.rstrip('@').split('@'))
    v_j_use = [np.expand_dims(v_j_one_hot,0)]
    b_j_use = [np.expand_dims(b_hot, 0)]
    #U_fea = [np.expand_dims(np.append(b_hot, s_j_vector),0)]
    q = DQN.main_predict(v=v_j_use,b=b_j_use)[0]
    recom = sorted(range(len(q)), key=lambda k: q[k], reverse=True)
    recom_print = [cate_reverse_dict[i] for i in recom]
    return recom_print

def simulate_test(DQN,Length,t):
    #b2 = DQN.sess.run(DQN.variable_list_main[1])
    #b1 = DQN.sess.run(DQN.variable_list_main[1])
    v_j_batch, v_j1_batch, Bias_one_hot, r_batch, click_batch, a_batch_one_hot, a_batch = transfer_data(Length,t)
    q = DQN.main_predict(v=v_j_batch,b=Bias_one_hot)
    sorted_recom = [sorted(range(len(each)),key=lambda x:each[x],reverse=True) for each in q]
    recom_index = [{val:index for index,val in enumerate(each)} for each in sorted_recom]
    mark1 = [subcat - recom_index[i][action] if click_batch[i] == 1 else  recom_index[i][action] + 1 for i,action in enumerate(a_batch)]
    mark2 = [subcat - recom_index[i][action]  for i, action in enumerate(a_batch) if click_batch[i] == 1 ]
    #mark = [1 if r_batch[i] == r_click and recom_index[i][action] == 0 else 0 for i, action in enumerate(a_batch)]
    return 1.0* sum(mark1) / (1.0*ACTIONS* max(len(mark1),1)),1.0* sum(mark2) / (1.0*ACTIONS* max(len(mark2),1))


def train(DQN,sess):
    # define the cost function

    # store the previous observations in replay memory


    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    checkpoint = tf.train.get_checkpoint_state("saved_networks_test")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    #epsilon = INITIAL_EPSILON
    Loss_along = []

    #initialize target from all zero
    DQN.train_target()
    while True:
        # choose an action epsilon greedily
            # sample a minibatch to train on
        t = tf.train.global_step(sess, global_step)
        v_j_batch, v_j1_batch, Bias_one_hot, r_batch, click_batch, a_batch_one_hot, a_batch\
            = transfer_data(BATCH,t)
        y_batch = []
        target_q_predict = DQN.target_predict(v=v_j1_batch,b=Bias_one_hot)
        for i in range(len(r_batch)):
            #if click_batch[i] == -1:
                y_batch.append(r_batch[i] + GAMMA * np.max(target_q_predict[i]))
            #else:
                #y_batch.append(r_batch[i])
        # perform gradient step

        Loss_here = DQN.train_main(a_batch_one_hot, y_batch, v=v_j_batch,b=Bias_one_hot)
        DQN.train_target()

        #B1 = sess.run(DQN.variable_list_main[1])
        #print("TIMESTEP", tf.train.global_step(sess, global_step), 'B1 ', B1)
        #print("TIMESTEP", tf.train.global_step(sess, global_step), 'DQN.main_predict ', DQN.main_predict([s_j_batch[0]]))
        Loss_along.append(Loss_here)
        #print("TIMESTEP" + str(tf.train.global_step(sess, global_step)), 'max(y_batch)', max(y_batch))
        #print("TIMESTEP" + str(tf.train.global_step(sess, global_step)), 'ave(y_batch)', sum(y_batch)/len(y_batch))
        print("TIMESTEP", tf.train.global_step(sess, global_step), 'Loss ', Loss_here)

        if int(tf.train.global_step(sess, global_step)) % 2000 == 0:
            saver.save(sess, 'saved_networks_test/' + GAME + '-dqn', global_step = global_step)
            #n = random.randint(0,9)
            #f = open(data_prefix + str(n), 'r')
            #global D
            #D = f.readlines()
            #f.close()
        if int(tf.train.global_step(sess, global_step)) % 200 == 0:

            #print()

            #b = sess.run([DQN.variable_list_main[3]])
            #print("TIMESTEP"+str(tf.train.global_step(sess, global_step)),'b',b)

            loss_ave = float(sum(Loss_along)) / max(len(Loss_along), 1)
            with open(Loss_file, 'a') as Loss_f:
                Loss_f.write("TIMESTEP"+str(tf.train.global_step(sess, global_step)) + '\t' + str(loss_ave) + '\n')
            Loss_along = []
            user1_bias = '25-34@男@高中及以下'
            user1_like = '游戏'
            user1_dislike = '汽车@汽车'
            print('potential',potential_part(user1_bias, user1_like+'@'+user1_dislike, '影视'+'@'+user1_dislike))

            user2_bias = '45-54@女@高中及以下'
            user2_like = '舞蹈'
            user2_dislike = '游戏@军事'

            recom1 = simulate_recom(DQN,user1_like,user1_dislike,user1_bias)
            recom2 = simulate_recom(DQN,user2_like,user2_dislike,user2_bias)


            v = [[one_hot_state('游戏@@'.split("@")).tolist()]]
            b = [[one_hot_bias('@@').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            print('游戏_max',max(q_game))
            recom = sorted(range(len(q_game)), key=lambda k: q_game[k], reverse=True)
            recom_print = [cate_reverse_dict[i] for i in recom]
            print('游戏_like    ', '\t'.join(recom_print))


            v = [[one_hot_state('@游戏@游戏'.split("@")).tolist()]]
            b = [[one_hot_bias('@@').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            #print('\t'.join(['--'.join(map(str, each)) for each in zip(q_game, cate_list)]))

            recom = sorted(range(len(q_game)), key=lambda k: q_game[k], reverse=True)
            recom_print = [cate_reverse_dict[i] for i in recom]
            #print('游戏_dislike ', '\t'.join(recom_print))

            v = [[one_hot_state('影视@@'.split("@")).tolist()]]
            b = [[one_hot_bias('@@').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            print('影视_max', max(q_game))
            print('\t'.join(['--'.join(map(str, each)) for each in zip(q_game, cate_list)]))

            v = [[one_hot_state('娱乐@@'.split("@")).tolist()]]
            b = [[one_hot_bias('@@').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            print('娱乐_max', max(q_game))

            v = [[one_hot_state('大自然@@'.split("@")).tolist()]]
            b = [[one_hot_bias('@@').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            print('大自然_max', max(q_game))
            #print('\t'.join(['--'.join(map(str, each)) for each in zip(q_game, cate_list)]))

            v = [[one_hot_state('舞蹈@@'.split("@")).tolist()]]
            b = [[one_hot_bias('@@').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            print('舞蹈_max', max(q_game))
            #print('\t'.join(['--'.join(map(str, each)) for each in zip(q_game, cate_list)]))

            recom = sorted(range(len(q_game)), key=lambda k: q_game[k], reverse=True)
            recom_print = [cate_reverse_dict[i] for i in recom]
            print('舞蹈_like    ', '\t'.join(recom_print))


            v = [[one_hot_state('@舞蹈@舞蹈'.split("@")).tolist()]]
            b = [[one_hot_bias('@@').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            #print('\t'.join(['--'.join(map(str, each)) for each in zip(q_game, cate_list)]))

            recom = sorted(range(len(q_game)), key=lambda k: q_game[k], reverse=True)
            recom_print = [cate_reverse_dict[i] for i in recom]
            #print('舞蹈_dislike ', '\t'.join(recom_print))


            v = [[one_hot_state('@@'.split("@")).tolist()]]
            b = [[one_hot_bias('25-34@男@高中及以下').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            recom = sorted(range(len(q_game)), key=lambda k: q_game[k], reverse=True)
            recom_print = [cate_reverse_dict[i] for i in recom]
            print('user1b ', '\t'.join(recom_print))

            v = [[one_hot_state('||'.split("|")).tolist()]]
            b = [[one_hot_bias('45-54@女@高中及以下').tolist()]]
            q_game = DQN.main_predict(np.array(v),np.array(b))[0]
            recom = sorted(range(len(q_game)), key=lambda k: q_game[k], reverse=True)
            recom_print = [cate_reverse_dict[i] for i in recom]
            print('user2b ', '\t'.join(recom_print))


            print('user1 : '+'\t'.join(recom1))
            print('user2 : ' + '\t'.join(recom2))
            #with open(Recom_file, 'a') as Recom_f:
                #Recom_f.write("TIMESTEP" + str(tf.train.global_step(sess, global_step)) + '  user1 : ' + '\t'.join(recom1) + '\n')
                #Recom_f.write("TIMESTEP" + str(tf.train.global_step(sess, global_step)) + '  user2 : ' + '\t'.join(recom2) + '\n')
            auc1,auc2 = simulate_test(DQN,500,t)
            print('like_pos',auc2)
            print('whole_pos', auc1)
            #with open(Auc_file, 'a') as auc_f:
               # auc_f.write("TIMESTEP" + str(tf.train.global_step(sess, global_step))  + '\t'+ str(auc) + '\n')


        #print("TIMESTEP",tf.train.global_step(sess, global_step), "/ STATE", state, 'Loss ',Loss_here)
        # write info to files

def playGame():
    with tf.Session() as sess:
        DQN = Network(sess,DUELING)
        train(DQN,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()


