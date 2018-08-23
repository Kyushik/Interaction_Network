# Cartpole
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Import modules
import tensorflow as tf
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
import time
import gym

env = gym.make('CartPole-v0')
game_name = 'CartPole'
algorithm = 'DQN'

# Parameter setting
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.01

Num_replay_memory = 10000
Num_start_training = 10000
Num_training = 15000
Num_testing  = 10000
Num_update = 150
Num_batch = 32
Num_episode_plot = 20

Is_render = True

# Initialize weights and bias
def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

def bias_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

# Assigning network variables to target network variables
def assign_network_to_target():
    # Get trainable variables
    trainable_variables = tf.trainable_variables()
    # network variables
    trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

    # target variables
    trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

    for i in range(len(trainable_variables_network)):
        sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

def interaction_net(O,Rr,Rs,Ra,X, output_len):

    object_state = O.get_shape()[-2]
    object_num = O.get_shape()[-1]
    
    relation_num = Rr.get_shape()[-1]
    
    B = tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra], axis = 1)
    
    B_len = B.get_shape()[-2]
    
# fR fully connected layer의 weight와 bias 정의하는 부분
# hidden num = 1개의 layer 안에 있는 노드의 수
# hidden_layer = hidden layer의 수

    fR_hidden_num = 512
    fR_hidden_layer = 3
    
    fR_weight = [weight_variable('fR_w_h'+str(i), [fR_hidden_num, fR_hidden_num]) for i in range(fR_hidden_layer-1)]
    fR_weight.insert(0, weight_variable('fR_w_input', [B_len, fR_hidden_num]))
    fR_bias = [bias_variable('fR_b_h'+str(i), [fR_hidden_num]) for i in range(fR_hidden_layer-1)]
    fR_bias.insert(0, bias_variable('fR_b_input',[fR_hidden_num]))
    
    e_list = []
    
    for i in range(relation_num):
        temp_B = tf.reshape(tf.slice(B,[0,0,i],[-1,-1,1]),[-1, B_len])
        for layer in range(fR_hidden_layer):
            if layer == 0:
                fR_hidden_state = tf.nn.relu(tf.matmul(temp_B, fR_weight[layer]) + fR_bias[layer])
            else:
                fR_hidden_state = tf.nn.relu(tf.matmul(fR_hidden_state, fR_weight[layer]) + fR_bias[layer])
        
        e_list.append(fR_hidden_state)
        
    E = tf.stack(e_list, axis = 2)
    E_bar = tf.matmul(E, Rr, transpose_b = True)
    
    C = tf.concat([O, X, E_bar], axis = 1)
    
    C_len = C.get_shape()[-2]
    
# fO fully connected layer의 weight와 bias 정의하는 부분
    
    fO_hidden_num = 512
    fO_hidden_layer = 3
    
    fO_weight = [weight_variable('fO_w_h'+str(i), [fO_hidden_num, fO_hidden_num]) for i in range(fO_hidden_layer-1)]
    fO_weight.insert(0, weight_variable('fO_w_input', [C_len, fO_hidden_num]))
    fO_bias = [bias_variable('fO_b_h'+str(i), [fO_hidden_num]) for i in range(fO_hidden_layer-1)]
    fO_bias.insert(0, bias_variable('fO_b_input',[fO_hidden_num]))
    
    output_weight = weight_variable('out_w', [fO_hidden_num, output_len])
    output_bias = weight_variable('out_b', [output_len])
    
    out_list = []
    
    for i in range(object_num):
        temp_C = tf.reshape(tf.slice(C,[0,0,i],[-1,-1,1]),[-1, C_len])
        for layer in range(fO_hidden_layer):
            if layer == 0:
                fO_hidden_state = tf.nn.relu(tf.matmul(temp_C, fO_weight[layer]) + fO_bias[layer])
            else:
                fO_hidden_state = tf.nn.relu(tf.matmul(fO_hidden_state, fO_weight[layer]) + fO_bias[layer])
                
        temp_out = tf.matmul(fO_hidden_state, output_weight) + output_bias
        out_list.append(temp_out)
    
    output = tf.stack(out_list, axis = 2)
    
    print(output)        
    
    return output


# Input
x = tf.placeholder(tf.float32, shape = [None, 4])

# IN layer variables

batch_size = tf.shape(x)[0]

object_num = 1
relation_num = 1
relation_state = 1
external_state = 1

x_reshape = tf.reshape(x, [-1, 4, object_num])

Rr = tf.ones([batch_size, object_num, relation_num])
Rs = tf.ones([batch_size, object_num, relation_num])
Ra = tf.zeros([batch_size, relation_state, relation_num])
external = tf.zeros([batch_size, external_state, object_num])

with tf.variable_scope('network'):
    output = interaction_net(O = x_reshape, Rr = Rr, Rs = Rs, Ra = Ra, X = external, output_len = Num_action)

output = tf.reshape(output, [-1, Num_action])

# Densely connect layer variables target
with tf.variable_scope('target'): 
    output_target = interaction_net(O = x_reshape, Rr = Rr, Rs = Rs, Ra = Ra, X = external, output_len = Num_action)

output_target = tf.reshape(output_target, [-1, Num_action])

# Loss function and Train
action_loss = tf.placeholder(tf.float32, shape = [None, Num_action])
y_target = tf.placeholder(tf.float32, shape = [None])

y_prediction = tf.reduce_sum(tf.multiply(output, action_loss), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Initial parameters
Replay_memory = []

step = 1
score = 0
episode = 0

plot_y_loss = []
plot_y_maxQ = []
loss_list = []
maxQ_list = []

data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

state = env.reset()

# Figure and figure data setting
plot_x = []
plot_y = []

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

progress = 'Exploring'
# Making replay memory
while True:
    if Is_render and progress =='Testing':
        # Rendering
        env.render()

    if step <= Num_start_training:
        progress = 'Exploring'
    elif step <= Num_start_training + Num_training:
        progress = 'Training'
    elif step < Num_start_training + Num_training + Num_testing:
        progress = 'Testing'
    else:
        # Test is finished
        print('Test is finished!!')
        plt.savefig('./plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')
        break

    # Select Action (Epsilon Greedy)
    if random.random() < Epsilon:       
        action = np.zeros([Num_action])
        action[random.randint(0, Num_action - 1)] = 1.0
        action_step = np.argmax(action)
    else:
        Q_value = output.eval(feed_dict={x: [state]})[0]
        action = np.zeros([Num_action])
        action[np.argmax(Q_value)] = 1
        action_step = np.argmax(action)

    state_next, reward, terminal, info = env.step(action_step)

    if progress != 'Testing':
        # Training to stay at the center 
        reward -= 5 * abs(state_next[0])

    # Save experience to the Replay memory
    if len(Replay_memory) > Num_replay_memory:
        del Replay_memory[0]

    Replay_memory.append([state, action, reward, state_next, terminal])

    if progress == 'Training':
        minibatch =  random.sample(Replay_memory, Num_batch)

        # Save the each batch data
        state_batch      = [batch[0] for batch in minibatch]
        action_batch     = [batch[1] for batch in minibatch]
        reward_batch     = [batch[2] for batch in minibatch]
        state_next_batch = [batch[3] for batch in minibatch]
        terminal_batch 	 = [batch[4] for batch in minibatch]

        y_batch = []

        # Update target network according to the Num_update value
        if step % Num_update == 0:
            assign_network_to_target()

        # Get y_prediction
        Q_batch = output_target.eval(feed_dict = {x: state_next_batch})
        for i in range(len(minibatch)):
            if terminal_batch[i] == True:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[i]))

        loss, _ = sess.run([Loss, train_step], feed_dict = {action_loss: action_batch, 
                                                            y_target: y_batch, 
                                                            x: state_batch})

        loss_list.append(loss)
        maxQ_list.append(np.max(Q_batch))

        # Reduce epsilon at training mode
        if Epsilon > Final_epsilon:
            Epsilon -= 1.0/Num_training

    if progress == 'Testing':
        Epsilon = 0

    # Update parameters at every iteration
    step += 1
    score += reward
    state = state_next

    # Plot average score
    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and progress != 'Exploring':
        ax1.plot(np.average(plot_x), np.average(plot_y_loss), '*')
        ax1.set_title('Mean Loss')
        ax1.set_ylabel('Mean Loss')
        ax1.hold(True)

        ax2.plot(np.average(plot_x), np.average(plot_y),'*')
        ax2.set_title('Mean score')
        ax2.set_ylabel('Mean score')
        ax2.hold(True)

        ax3.plot(np.average(plot_x), np.average(plot_y_maxQ),'*')
        ax3.set_title('Mean Max Q')
        ax3.set_ylabel('Mean Max Q')
        ax3.set_xlabel('Episode')
        ax3.hold(True)

        plt.draw()
        plt.pause(0.000001)

        plot_x = []
        plot_y = []
        plot_y_loss = []
        plot_y_maxQ = []

    # Terminal
    if terminal == True:
        print('step: ' + str(step) + ' / '  + 
              'episode: ' + str(episode) + ' / ' +
              'progess: ' + progress  + ' / '  + 
              'epsilon: ' + str(Epsilon) + ' / '  + 
              'score: ' + str(score))

        if progress != 'Exploring':
            # add data for plotting
            plot_x.append(episode)
            plot_y.append(score)
            plot_y_loss.append(np.mean(loss_list))
            plot_y_maxQ.append(np.mean(maxQ_list))

        score = 0
        loss_list = []
        maxQ_list = []
        episode += 1

        state = env.reset()