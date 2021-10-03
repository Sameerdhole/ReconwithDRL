import tensorflow as tf
import numpy as np
from network.loss_functions import huber_loss, mse_loss , kl_loss
from network.network import C3F2, C3F2_ActorCriticShared, C3F2_Actor, C3F2_Critic
from numpy import linalg as LA
#from tensorflow.python.keras._impl.keras.losses import kullback_leibler_divergence as kd
from tensorflow.losses import KLDivergence as kld

###########################################################################
# DeepPPO: Class
###########################################################################

class initialize_network_DeepPPO():
    def __init__(self, cfg, name, vehicle_name):
        self.g = tf.Graph()
        self.vehicle_name = vehicle_name
        self.iter_baseline = 0
        self.iter_policy = 0
        self.first_frame = True
        self.last_frame = []
        self.iter_combined = 0
        with self.g.as_default():
            stat_writer_path = cfg.network_path + self.vehicle_name + '/return_plot/'
            loss_writer_path = cfg.network_path + self.vehicle_name + '/loss' + name + '/'
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            # name_array = 'D:/train/loss'+'/'+name
            self.loss_writer = tf.summary.FileWriter(loss_writer_path)
            self.env_type = cfg.env_type
            self.input_size = cfg.input_size
            self.num_actions = cfg.num_actions
            self.eps_clip = cfg.eps_clip

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

            # self.X = tf.image.resize_images(self.X1, (227, 227))

            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            # self.target = tf.placeholder(tf.float32, shape=[None], name='action_probs')
            # self.target_baseline = tf.placeholder(tf.float32, shape=[None], name='baseline')
            self.actions = tf.placeholder(tf.int32, shape=[None, 1], name='Actions')
            self.TD_target = tf.placeholder(tf.float32, shape=[None, 1], name='TD_target')
            self.prob_old = tf.placeholder(tf.float32, shape=[None, 1], name='prob_old')
            self.GAE = tf.placeholder(tf.float32, shape=[None, 1], name='GAE')

            # Select the deep network
            self.model = C3F2_ActorCriticShared(self.X, cfg.num_actions, cfg.train_fc)
            self.pi = self.model.action_probs
            self.state_value = self.model.state_value

            self.ind = tf.one_hot(tf.squeeze(self.actions), cfg.num_actions)
            self.pi_a = tf.expand_dims(tf.reduce_sum(tf.multiply(self.pi, self.ind), axis=1), axis=1)

            self.ratio = tf.exp(tf.log(self.pi_a+1e-10) - tf.log(self.prob_old+1e-10))
            p1 = self.ratio * self.GAE
            p2 = tf.clip_by_value(self.ratio, 1-self.eps_clip, 1+self.eps_clip)*self.GAE

            # Define losses
            self.loss_actor_op = -tf.reduce_mean(tf.minimum(p1, p2))
            self.loss_entropy = 0.5*tf.reduce_mean(tf.multiply((tf.log(self.pi) + 1e-8), self.pi))
            self.loss_critic_op = 0.5*mse_loss(self.state_value, self.TD_target)

            self.loss_op = self.loss_critic_op + self.loss_actor_op + self.loss_entropy

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss_op, name="train_main")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()

            self.sess.graph.finalize()

        # Load custom weights from custom_load_path if required
        if cfg.custom_load:
            print('Loading weights from: ', cfg.custom_load_path)
            self.load_network(cfg.custom_load_path)

    def get_vars(self):
        return self.sess.run(self.all_vars)

    def initialize_graphs_with_average(self, agent, agent_on_same_network):
        values = {}
        var = {}
        all_assign = {}
        for name_agent in agent_on_same_network:
            values[name_agent] = agent[name_agent].network_model.get_vars()
            var[name_agent] = agent[name_agent].network_model.all_vars
            all_assign[name_agent] = []

        for i in range(len(values[name_agent])):
            val = []
            for name_agent in agent_on_same_network:
                val.append(values[name_agent][i])
            # Take mean here
            mean_val = np.average(val, axis=0)
            for name_agent in agent_on_same_network:
                # all_assign[name_agent].append(tf.assign(var[name_agent][i], mean_val))
                var[name_agent][i].load(mean_val, agent[name_agent].network_model.sess)

    def prob_actions(self, xs):
        TD_target = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        prob_old = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        GAE = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.pi,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                        self.actions: actions,
                                        self.TD_target: TD_target,
                                        self.prob_old: prob_old,
                                        self.GAE: GAE})


    def get_state_value(self, xs):
        lr = 0
        actions = np.zeros(dtype=int, shape=[xs.shape[0], 1])
        TD_target = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        prob_old = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        GAE = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)

        baseline = self.sess.run(self.state_value,
                                 feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                            self.X1: xs,
                                            self.actions: actions,
                                            self.TD_target: TD_target,
                                            self.prob_old: prob_old,
                                            self.GAE: GAE})
        return baseline

    def train_policy(self, xs, actions, TD_target, prob_old, GAE, lr, iter):
        self.iter_policy += 1
        batch_size = xs.shape[0]
        train_eval = self.train_op
        loss_eval = self.loss_op
        predict_eval = self.pi

        _, loss, loss_critic, loss_actor,loss_entropy, ProbActions = self.sess.run([train_eval, loss_eval, self.loss_critic_op, self.loss_actor_op, self.loss_entropy, predict_eval],
                                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                        self.X1: xs,
                                                        self.actions: actions,
                                                        self.TD_target: TD_target,
                                                        self.prob_old: prob_old,
                                                        self.GAE: GAE})
        

        #B.append(state_value)
        #b.append(td_target)


        MaxProbActions = np.max(ProbActions)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss_Total', group=self.vehicle_name, value=LA.norm(loss) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Actor', group=self.vehicle_name, value=LA.norm(loss_actor) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Critic', group=self.vehicle_name, value=LA.norm(loss_critic) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Entropy', group=self.vehicle_name, value=LA.norm(loss_entropy) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=self.iter_policy)
        self.log_to_tensorboard(tag='MaxProb', group=self.vehicle_name, value=MaxProbActions, index=self.iter_policy)

    def action_selection_with_prob(self, state):
        action = np.zeros(dtype=int, shape=[state.shape[0], 1])
        prob_action = np.zeros(dtype=float, shape=[state.shape[0], 1])

        probs = self.sess.run(self.pi,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.actions: action})

        for j in range(probs.shape[0]):
            action[j] = np.random.choice(self.num_actions, 1, p=probs[j])[0]
            prob_action[j] = probs[j][action[j][0]]

        return action.astype(int), prob_action,probs

    def action_selection(self, state):
        action = np.zeros(dtype=int, shape=[state.shape[0], 1])
        prob_action = np.zeros(dtype=float, shape=[state.shape[0], 1])

        probs = self.sess.run(self.pi,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.actions: action})

        for j in range(probs.shape[0]):
            action[j] = np.random.choice(self.num_actions, 1, p=probs[j])[0]
            prob_action[j] = probs[j][action[j][0]]

        return action.astype(int)

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, save_path, episode=''):
        save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        self.saver.save(self.sess, save_path)
        print('Model Saved: ', save_path)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)



###########################################################################
# DeepPPG: Class
###########################################################################

class initialize_network_DeepPPG():
    def __init__(self, cfg, name, vehicle_name):
        self.g = tf.Graph()
        self.vehicle_name = vehicle_name
        self.iter_baseline = 0
        self.iter_policy = 0
        self.first_frame = True
        self.last_frame = []
        self.iter_combined = 0
        with self.g.as_default():
            stat_writer_path = cfg.network_path + self.vehicle_name + '/return_plot/'
            loss_writer_path = cfg.network_path + self.vehicle_name + '/loss' + name + '/'
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            # name_array = 'D:/train/loss'+'/'+name
            self.loss_writer = tf.summary.FileWriter(loss_writer_path)
            self.env_type = cfg.env_type
            self.input_size = cfg.input_size
            self.num_actions = cfg.num_actions
            self.eps_clip = cfg.eps_clip
            self.beta=cfg.beta
            self.beta_s=cfg.beta_s

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

            # self.X = tf.image.resize_images(self.X1, (227, 227))

            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            # self.target = tf.placeholder(tf.float32, shape=[None], name='action_probs')
            # self.target_baseline = tf.placeholder(tf.float32, shape=[None], name='baseline')
            self.actions = tf.placeholder(tf.int32, shape=[None, 1], name='Actions')
            self.TD_target = tf.placeholder(tf.float32, shape=[None, 1], name='TD_target')

            

            self.prob_old = tf.placeholder(tf.float32, shape=[None, 1], name='prob_old')
            self.GAE = tf.placeholder(tf.float32, shape=[None, 1], name='GAE')
            self.D_target = tf.placeholder(tf.float32, shape=[None, 1], name='D_target')
            # Select the deep network

            self.model_pi = C3F2_Actor(self.X,cfg.num_actions,cfg.train_fc)
            self.model_v = C3F2_Critic(self.X,cfg.num_actions,cfg.train_fc)
            #self.model = C3F2_ActorCriticShared(self.X, cfg.num_actions, cfg.train_fc)
            ##can decouple actor and critic here based on models.
            ##get_state values feeds params to model.state_value which in turn calls c3f2 model and returns state value 
            self.pi = self.model_pi.action_probs
            self.state_value = self.model_v.state_value

            self.old_pi = self.pi

            self.ind = tf.one_hot(tf.squeeze(self.actions), cfg.num_actions)
            self.pi_a = tf.expand_dims(tf.reduce_sum(tf.multiply(self.pi, self.ind), axis=1), axis=1)

            self.ratio = tf.exp(tf.log(self.pi_a+1e-10) - tf.log(self.prob_old+1e-10))
            p1 = self.ratio * self.GAE
            p2 = tf.clip_by_value(self.ratio, 1-self.eps_clip, 1+self.eps_clip)*self.GAE

            # Define losses
            #Lclip loss
            self.loss_actor_op = -tf.reduce_mean(tf.minimum(p1, p2))

            #S_pi entropy loss
            self.loss_entropy = 0.5*tf.reduce_mean(tf.multiply((tf.log(self.pi) + 1e-8), self.pi))

            #E_Pi loss
            self.L_pi=self.loss_actor_op+tf.multiply(self.loss_entropy ,self.beta_s)
            
            #L aux auxilary objective/L_value
            self.L_aux = 0.5*mse_loss(self.state_value, self.TD_target)

            #Value Loss
            self.L_v = mse_loss(0.707*self.state_value, 0.707*self.TD_target)
            
            #KL Loss(yet to be written correctly)
            self.kl=kl_loss(self.pi, self.old_pi)
            
            #Ljoint
            self.loss_op = self.L_aux +tf.multiply(float(self.beta),self.kl)
            
            #OPTIMIZERS
            
            #PPO OP 
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss_op, name="train_main")
            
            #Policy(E_pi) OP
            self.E_pi_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.L_pi, name="train_main")
            
            #Value(E_v) OP
            self.E_v_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.L_v, name="train_main")

            #Value(E_aux) OP
            self.E_aux_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.L_aux, name="train_main")
            
            #L_joint OP
            self.E_joint_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss_op, name="train_main")


            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()

            self.sess.graph.finalize()



    def get_vars(self):
        return self.sess.run(self.all_vars)

    def initialize_graphs_with_average(self, agent, agent_on_same_network):
        values = {}
        var = {}
        all_assign = {}
        for name_agent in agent_on_same_network:
            values[name_agent] = agent[name_agent].network_model.get_vars()
            var[name_agent] = agent[name_agent].network_model.all_vars
            all_assign[name_agent] = []

        for i in range(len(values[name_agent])):
            val = []
            for name_agent in agent_on_same_network:
                val.append(values[name_agent][i])
            # Take mean here
            mean_val = np.average(val, axis=0)
            for name_agent in agent_on_same_network:
                # all_assign[name_agent].append(tf.assign(var[name_agent][i], mean_val))
                var[name_agent][i].load(mean_val, agent[name_agent].network_model.sess)      

    def prob_actions(self, xs):
        TD_target = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        prob_old = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        GAE = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.pi,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                        self.actions: actions,
                                        self.TD_target: TD_target,
                                        self.prob_old: prob_old,
                                        self.GAE: GAE})


    def get_state_value(self, xs):
        lr = 0
        actions = np.zeros(dtype=int, shape=[xs.shape[0], 1])
        TD_target = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        prob_old = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        GAE = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)

        baseline = self.sess.run(self.state_value,
                                 feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                            self.X1: xs,
                                            self.actions: actions,
                                            self.TD_target: TD_target,
                                            self.prob_old: prob_old,
                                            self.GAE: GAE})
        return baseline
    



    def action_selection_with_prob(self, state):
        action = np.zeros(dtype=int, shape=[state.shape[0], 1])
        prob_action = np.zeros(dtype=float, shape=[state.shape[0], 1])
        self.old_pi = self.pi
        probs = self.sess.run(self.pi,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.actions: action})

        for j in range(probs.shape[0]):
            action[j] = np.random.choice(self.num_actions, 1, p=probs[j])[0]
            prob_action[j] = probs[j][action[j][0]]

        return action.astype(int), prob_action,probs


    def action_selection(self, state):
        action = np.zeros(dtype=int, shape=[state.shape[0], 1])
        prob_action = np.zeros(dtype=float, shape=[state.shape[0], 1])
        self.old_pi = self.pi
        probs = self.sess.run(self.pi,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.actions: action})

        for j in range(probs.shape[0]):
            action[j] = np.random.choice(self.num_actions, 1, p=probs[j])[0]
            prob_action[j] = probs[j][action[j][0]]

        return action.astype(int)



    def train_policy(self, xs, actions, TD_target, prob_old, GAE, lr, iter,E_pi,E_v):
        self.iter_policy += 1
        batch_size = xs.shape[0]
        train_eval = self.train_op
        L_pi_eval=self.E_pi_op
        loss_eval = self.loss_op
        predict_eval = self.pi
        predict_state = self.state_value
        L_v_eval=self.E_v_op


        #optimize L_pi
        for m in range(E_pi):
            _,L_pi,L_clip,L_entrop, ProbActions = self.sess.run([L_pi_eval,self.L_pi,self.loss_actor_op,self.loss_entropy , predict_eval],
                                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                        self.X1: xs,
                                                        self.actions: actions,
                                                        self.TD_target: TD_target,
                                                        self.prob_old: prob_old,
                                                        self.GAE: GAE})

        #optimize L_v
        for n in range(E_v):
            _,L_v,state_value = self.sess.run([L_v_eval,self.L_v,predict_state],
                                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                        self.X1: xs,
                                                        self.actions: actions,
                                                        self.TD_target: TD_target,
                                                        self.prob_old: prob_old,
                                                        self.GAE: GAE})
            

        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss_Total', group=self.vehicle_name, value=LA.norm(L_pi) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Actor', group=self.vehicle_name, value=LA.norm(L_v) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Critic', group=self.vehicle_name, value=LA.norm(L_clip) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Entropy', group=self.vehicle_name, value=LA.norm(L_entrop) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=self.iter_policy)
        #self.log_to_tensorboard(tag='MaxProb', group=self.vehicle_name, value=MaxProbActions, index=self.iter_policy)


    
    

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, save_path, episode=''):
        save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        self.saver.save(self.sess, save_path)
        print('Model Saved: ', save_path)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)    

    def train_aux(self,xs, actions, TD_target, prob_old, GAE, lr):
        ##init params
        ##feeddict to optimize l joint
        ##Optimize L joint wrt theta_pi and theta_v
        #print(" inside network_model train_aux")

        self.iter_policy += 1
        batch_size = xs.shape[0]
        joint_eval = self.E_joint_op
        train_eval = self.train_op
        loss_eval = self.loss_op
        predict_eval = self.pi
        predict_state = self.state_value
        L_v_eval=self.E_v_op
        #print('for loop1')
        #optimize L

        _,L_a,L_o,L_kl, ProbActions = self.sess.run([joint_eval,self.L_aux,self.loss_op,self.kl , predict_eval],
                                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                        self.X1: xs,
                                                        self.actions: actions,
                                                        self.TD_target: TD_target,
                                                        self.prob_old: prob_old,
                                                        self.GAE: GAE})
        print('for loop2')
        #optimize L_v
        _,L_v,state_value = self.sess.run([L_v_eval,self.L_v,predict_state],
                                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                        self.X1: xs,
                                                        self.actions: actions,
                                                        self.TD_target: TD_target,
                                                        self.prob_old: prob_old,
                                                        self.GAE: GAE})

        print(L_kl)
        print(L_kl.shape)
        print(self.pi)
        print(self.old_pi)
        print(self.pi.shape)

"""    def update_beta():
        if(self.kl<self.D_target/1.5 ):
            self.beta=selfbeta/2
        elif(self.kl>self.D_target/1.5 ):
            self.beta=selfbeta*2
"""