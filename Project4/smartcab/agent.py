import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here 
        self.Q_table=[[0 for i in range(4)] for i in range(2*4**4)]   #There are 512 kinds of states, so the Q_table size of 512*4 
        self.total_score=0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def state_index(self,inputs,next_waypoint):
        '''
        return the index of a specific state, which will be used in Q_table
        '''
        light=['green','red']
        actions=self.env.valid_actions
        a=light.index(inputs['light'])
        b=actions.index(inputs['oncoming'])
        c=actions.index(inputs['left'])
        d=actions.index(inputs['right'])
        e=actions.index(next_waypoint)
        return a*4**4+b*4**3+c*4**2+d*4+e        #return the index by 'quaternary system', because acitons has 4 items

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # TODO: Update state
        self.state=(inputs,self.next_waypoint)

        # TODO: Select action according to your policy
        valid_actions=self.env.valid_actions
        index=self.state_index(inputs,self.next_waypoint)        
        iter=random.choice([0,1,2,3])    
        
        #to find an action to maxium Q(state, action)
        Q_max=0
        for i in range(4):
            sum=self.Q_table[index][i]
            if sum > Q_max:
                sum=Q_max
                iter = i
        action=valid_actions[iter]

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_score+=reward

        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()     #update state after act the action
        inputs = self.env.sense(self)
        new_index=self.state_index(inputs,self.next_waypoint)       
        Q_next=0
        for i in range(4):
            sum=self.Q_table[new_index][i]
            if sum > Q_next:
                Q_next=sum
        self.Q_table[index][valid_actions.index(action)] = (1-LEARNING_RATE)*self.Q_table[index][valid_actions.index(action)] + (LEARNING_RATE)*( reward + GAMMA * Q_next )  #implemented by a fixed LEARNING_RATE

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=15)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  
    sim.run(n_trials=100)           #train the model, especially update Q_table
    
    #e.reach_end=0
    #a.total_score=0          
    
    #After training the agent, here try out to see the result
    sim2=Simulator(e,update_delay=0.5)        
    sim2.run(n_trials=10)			
    #f.writelines('%d LEARNING_RATE: %.2f  GAMMA: %.2f  Reach_end: %d  Total_score: %d\n'%(count,LEARNING_RATE,GAMMA,e.reach_end,a.total_score))


#log file, you may need to change here    
#f=open('C:/Users/haoran/Desktop/log.txt','a')
count=0      
def tune_parameters():
    '''
    trying and choose best LEARNING_RATE and GAMMA
    '''
    if __name__ == '__main__':
        for LEARNING_RATE in range(5,10):
            LEARNING_RATE/=10.0
            for GAMMA in range(5,10):
                GAMMA/=10.0
                run()
                count+=1
#tune_parameters()

#RESULT
LEARNING_RATE=0.7
GAMMA=0.8
if __name__=='__main__':
    run()

#f.close()
