import functools
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# RecSim imports
from recsim import agent
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
from recsim.simulator import runner_lib
from recsim.agent import AbstractEpisodicRecommenderAgent
from model import Actor,Critic, Noise,RelayBuffer
from enviroment import UserModel,LTSDocumentSampler,LTSStaticUserSampler,LTSResponse,clicked_engagement_reward
import pandas as pd
import os
import data_preprocess
from model import Actor,Critic
import os
import tensorflow as tf
from rank_metric import precision_at_k,ndcg_at_k
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Actor_Critic_Agent(AbstractEpisodicRecommenderAgent):
    def __init__(self, sess,
               observation_space,
               action_space,actor_model, critic_model,buffer,noise_model,slate_size,embedding_size,
               optimizer_name='',
               eval_mode=False,
               **kwargs):
    # Check if document corpus is large enough.
        if len(observation_space['doc'].spaces) < len(action_space.nvec):
            print("problem happens")
            raise RuntimeError('Slate size larger than size of the corpus.')
        super(Actor_Critic_Agent, self).__init__(action_space)

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.buffer = buffer
        self.noise_model = noise_model
        self.sess = sess
        self.slate_size = slate_size
        self.embedding_size = embedding_size

    def generate_doc_representation(self,doc):
        corpus = []
        origin_corpus = list(doc.values())
        for doc_obs in origin_corpus:
            doc_features = doc_obs['embedding_features']
            corpus.append(doc_features)
        corpus = np.array(corpus).reshape(len(origin_corpus), self.embedding_size)
        return corpus

    def generate_state_represetation(self,user):
        user_state = user.reshape(1,user.shape[0]*user.shape[1])
        return user_state

    def calculate_bellman(self,rewards,q_values,dones):
        return

    def step(self, reward, observation):
        """
        :param reward:
        :param observation:
        :return:
        """
        doc = observation['doc']
        user = observation['user']
        # user_id = user['user_id']
        # user_past_record_ids = user['record_ids']
        user_past_record = user['past_record']
        items_space = self.generate_doc_representation(doc)
        state = self.generate_state_represetation(user_past_record)
        actions = self.actor_model.predict(state)
        num_action_vector = actions.shape[1]//self.embedding_size
        actions = actions.reshape(num_action_vector,self.embedding_size)
        result_recommend_list = self.generate_actions(items_space,actions,topN=self.slate_size)
        return result_recommend_list

    def save_model(self,epoch,path=""):
        path = os.path.join(path,str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor_model.save(path)
        self.critic_model.save(path)

    def load_model(self,path=""):
        self.actor_model.load_model(path)
        # self.critic_model.load(path)

    def update_network_from_batches(self,batch_size):
        """

        :param batch_size:
        :return: best q-value in a batchs and critic loss
        """
        if self.buffer.size() < batch_size:
            return 0.0
        state_batch,action_batch,reward_batch,n_state_batch,terminal_batch = self.buffer.sample_batch(batch_size)

        # calculate predicted q value
        action_weights = self.actor_model.predict_target(state_batch)
        # n_action_batch = gene_actions(item_space, action_weights, action_len)
        # print("action weight shape : ",action_weights.shape)
        # print("next state batches shape : ",n_state_batch.shape)
        target_q_batch = self.critic_model.predict_target(n_state_batch, action_weights)
        y_batch = []
        for i in range(batch_size):
            y_batch.append(reward_batch[i] + self.critic_model.gamma * target_q_batch[i]*terminal_batch[i])
        # train critic
        target_critic_values = np.reshape(y_batch, (batch_size, 1))
        # print("target critic shape :",target_critic_values.shape)
        critic_loss = self.critic_model.train(state_batch, action_batch, target_critic_values)
        # q_value = self.critic_model.predict(state_batch, action_batch)
        # train actor
        action_weight_batch_for_gradients = self.actor_model.predict(state_batch)
        # action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients, action_len)
        a_gradient_batch = self.critic_model.gradients(state_batch,action_weight_batch_for_gradients)
        # print("a gradient batch : ",np.array(a_gradient_batch).reshape((-1, self.act_dim)))
        # self.actor_model.train(state_batch, a_gradient_batch[0])
        self.actor_model.train(state_batch, a_gradient_batch)
        # update target networks
        self.actor_model.update_target_network()
        self.critic_model.update_target_network()

        return critic_loss

    # def evaluate(self):

    def summarize(self,list_of_tag,epoch):
        """Log scalar variables."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in list_of_tag])
        self.writer.add_summary(summary, epoch)

    def train(self,max_episode,user_per_eps,batch_size,env,save_model_frequenly = 5,save_path = "",log_path = "logs/scalars/",history_path = "history_log"):
        summary_folder = log_path
        self.writer = tf.summary.FileWriter(summary_folder)

        slate_size = env.action_space.shape[0]

        # initialize target network weights
        self.actor_model.hard_update_target_network()
        self.critic_model.hard_update_target_network()
        actions_dim = (1,30)

        list_reward = []
        list_critic_loss = []
        list_precision_k = []
        # list_ndcg_k = list()
        for eps in range(max_episode):
            ep_reward = 0.
            ep_critic_loss = 0.
            ep_precision_k = 0
            # ep_ndcg_k = 0
            num_train_user = 0
            for u in range(user_per_eps):
                user_ep_reward = 0.
                user_ep_critic_loss = 0.
                user_ep_precision_k = 0
                # user_ep_ndcg_k = 0
                step = 0


                observation = env.reset()

                terminal = False
                doc = observation['doc']
                user = observation['user']
                user_past_record = user['past_record']
                items_space = self.generate_doc_representation(doc)
                state = self.generate_state_represetation(user_past_record)
                # print("current user : ", observation['user']['user_id'])

                if (len(doc.keys())<self.slate_size):
                    print("can not train with this user ")
                    terminal = True

                while(not terminal):
                # for step in range(episode_length):
                    actions = self.actor_model.predict(state)+self.noise_model.noise()
                    transform_action = actions.reshape(actions_dim)
                    result_recommend_list = self.generate_actions(items_space, transform_action,topN=self.slate_size)
                    # print("state obs : ", observation['user']['record_ids'])
                    # print("docs ids : ", observation['doc'].keys())
                    #
                    # print("recommended slate : ",result_recommend_list)

                    next_observation, reward, terminal, _ = env.step(result_recommend_list)

                    next_state = self.generate_state_represetation(next_observation['user']['past_record'])
                    # print("response : ", next_observation['response'])
                    # print("reward : ",reward)
                    # print("next state obs : ", next_observation['user']['record_ids'])
                    # print("total remaind candidate items to recommend : ", len(next_observation['doc'].keys()))
                    # print("docs ids : ", next_observation['doc'].keys())

                    #calculate the ndcg and precision
                    # rate_score_list = [response['rating'] for response in next_observation['response']]
                    click_list = [int(response['click']) for response in next_observation['response']]
                    precision_k = precision_at_k(click_list, slate_size)
                    # ndcg_k = ndcg_at_k(rate_score_list, slate_size, method=0)
                    user_ep_precision_k += precision_k
                    # user_ep_ndcg_k += ndcg_k

                    self.buffer.add(state.flatten(),actions.flatten(),reward,next_state.flatten(),terminal)
                    user_ep_reward += reward


                    critic_loss = self.update_network_from_batches(batch_size)
                    # user_ep_max_q_value += max_q_value
                    user_ep_critic_loss += critic_loss

                    #update observation
                    observation = next_observation
                    doc = observation['doc']
                    items_space = self.generate_doc_representation(doc)
                    state = next_state
                    step+=1

                if (step > 0):
                    # ep_max_q_value += (user_ep_max_q_value)/step
                    ep_critic_loss += (user_ep_critic_loss)/step
                    ep_reward += (user_ep_reward/step)
                    ep_precision_k +=user_ep_precision_k/step
                    # ep_ndcg_k +=user_ep_ndcg_k/step
                    num_train_user+=1

            if (num_train_user>0):
                # ep_max_q_value = ep_max_q_value/num_train_user
                ep_critic_loss = ep_critic_loss/num_train_user
                ep_reward = ep_reward/num_train_user
                ep_precision_k /= num_train_user
                # ep_ndcg_k /=num_train_user

            #add summary information on tensorboard for observation
            list_tag_values = {
                ("critic loss ",ep_critic_loss),
                # ("episode avg max q value",ep_max_q_value),
                ("episode avg reward ",ep_reward),
                ("episode avg precision k",ep_precision_k)
                # ("episode acg ndcgg k",ep_ndcg_k)
            }

            list_reward.append(ep_reward)
            list_critic_loss.append(ep_critic_loss)
            # list_max_q.append(ep_max_q_value)
            list_precision_k.append(ep_precision_k)
            # list_ndcg_k.append(ep_ndcg_k)

            self.summarize(list_tag_values,eps)
            if (eps+1)%save_model_frequenly ==0:
                print("update model ")
                path = save_path
                self.save_model(eps,path)
                history_report = {
                    "episode_reward": list_reward,
                    # "episode_max_q_value": list_max_q,
                    "episode_critic_loss": list_critic_loss,
                    "episode_precision_k": list_precision_k,
                    # "episode_ndcg_k": list_ndcg_k
                }
                history_table = pd.DataFrame(history_report)
                history_table.to_csv(os.path.join(history_path, "history_record.csv"), index=False)

            # if (j + 1) % 50 == 0:
            #     logger.info("=========={0} episode of {1} round: {2} reward=========".format(i, j, ep_reward))

            print("current episode : ",eps)
            print("eps reward : ",ep_reward)
            # print("eps q : ",ep_max_q_value)
            print("eps loss : ",ep_critic_loss)
            print("eps precision K ",ep_precision_k)
            # print("eps ndcg K ",ep_ndcg_k)
        #
        history_report = {
            "episode_reward":list_reward,
            # "episode_max_q_value":list_max_q,
            "episode_critic_loss": list_critic_loss,
            "episode_precision_k":list_precision_k,
            # "episode_ndcg_k": list_ndcg_k
        }
        return history_report

    def generate_actions(self,item_space, weight_batch,topN = 1):
        """use output of actor network to calculate action list
        Args:
            item_space: (m,n) where m is # of items and n is embedding size
            weight_batch: actor network outputs (k,n) where k is # of items to pick and n is embedding size
        Returns:
            recommendation list of index (k)
        """
        # print(item_space.shape)

        recommend_items = np.dot(weight_batch,np.transpose(item_space))
        if (recommend_items.shape[0] == 1):
            # print(recommend_items)
            return recommend_items[0].argsort()[::-1][:topN]
        else:
            # need to deal with multi actions
            # print("deal with multiple")
            result = list()
            for index in range(recommend_items.shape[0]):
                current_action_recommend_items = recommend_items[index]
                best_item_index = recommend_items.argsort()[::-1]
                result.append(best_item_index)
            return np.argmax(recommend_items, axis=1)

    # def begin_episode(self):
    #
    # def end_episode(self):
    #     hh
    #
    # def evaluate(self,):


