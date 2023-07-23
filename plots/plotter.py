import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def get_plots(parent_dir):
    check_dir = elegibility("train_dict.pkl",parent_dir)
    if check_dir:
        plot = Plot(parent_dir,"train_dict.pkl")
        plot.save_plot()
        plot.close()
    else:
        print('NO train plots')

    #for folder_name in os.listdir(parent_dir):
    #    dir = os.path.join(parent_dir,folder_name)
    #    check_dir = elegibility("plot_dict.pkl",dir)
    #    if check_dir:
    #        print(f' - {folder_name} -> YES')
    #        plot = Plot(dir,"plot_dict.pkl")
    #        
    #        plot.save_plot()
    #        plot.close()
    #        
    #        #frame = Frame(parent_dir,folder_name,"frames.pkl")
    #        #frame.save_plot()
    #    else:
    #        print(f' - {folder_name} -> NO')

def elegibility(name,folder):
    path_pkl = os.path.join(folder,name)  
    if not os.path.exists(path_pkl): 
        return False
    else:
        list_file = os.listdir(folder)
        for file in list_file:
            if file[-4:]=='.png':
                return False
    return True




class Plot():
    def __init__(self,dir,dict_name):
        self.dir = dir
        self.load_dict(dict_name)

    def save_plot(self,show=True):
        if self.type=='act':

            f_usr, axs = plt.subplots(2,1, sharey=True)
            axs[0].plot(self.timesteps,self.usr_cmd[:,0])
            axs[0].set_title('linear vel')
            axs[1].plot(self.timesteps,self.usr_cmd[:,1])
            axs[1].set_title('angular vel')
            path = os.path.join(self.dir,'usr_cmd.png')
            plt.savefig(path)

            f_ca, axs = plt.subplots(2,1, sharey=True)
            axs[0].plot(self.timesteps,self.ca_cmd[:,0])
            axs[0].set_title('linear vel')
            axs[1].plot(self.timesteps,self.ca_cmd[:,1])
            axs[1].set_title('angular vel')
            path = os.path.join(self.dir,'ca_cmd.png')
            plt.savefig(path)
            
            f_ts, axs = plt.subplots(2,1, sharey=True)
            axs[0].plot(self.timesteps,self.ts_cmd[:,0])
            axs[0].set_title('linear vel')
            axs[1].plot(self.timesteps,self.ts_cmd[:,1])
            axs[1].set_title('angular vel')
            path = os.path.join(self.dir,'ts_cmd.png')
            plt.savefig(path)

            f_com, axs = plt.subplots(2,1, sharey=True)
            axs[0].plot(self.timesteps,self.cmd[:,0])
            axs[0].set_title('linear vel')
            axs[1].plot(self.timesteps,self.cmd[:,1])
            axs[1].set_title('angular vel')
            path = os.path.join(self.dir,'commands.png')
            plt.savefig(path)

            plt.figure()
            try:
                plt.plot(self.timesteps,self.alpha,'r-')

            except:
                plt.plot(self.timesteps,self.alpha[:,0],'r-')
                plt.plot(self.timesteps,self.alpha[:,1],'b-')
                plt.plot(self.timesteps,self.alpha[:,2],'g-')
                plt.legend(['usr_a','ca_a','ts_a'])
            path = os.path.join(self.dir,'alpha.png')
            plt.savefig(path)
        
        else:
            plt.figure()
            plt.plot(self.epochs,self.rewards,'r-')
            path = os.path.join(self.dir,'rewards.png')
            plt.savefig(path)

            plt.figure()
            plt.plot(self.epochs,self.loss,'r-')
            path = os.path.join(self.dir,'loss.png')
            plt.savefig(path)

        print('Plots saved in: ',self.dir)
        self.close()
        #if not self.description=='':
        #    with open(os.path.join(self.dir,'description.txt'), mode='w') as f:
        #        f.write(self.description)

        if show:
            plt.show()

    def close(self):
        plt.close('all')


    def load_dict(self,dict_name):
        where = os.path.join(self.dir,dict_name)
        with open(where, 'rb') as handle:
            dict = pickle.load(handle)

        self.type = dict['type']
        print('type: ',self.type)
        if self.type=='act':
            self.timesteps=np.array(dict['timesteps'])
            self.usr_cmd=np.array(dict['usr_cmd'])
            self.ca_cmd=np.array(dict['ca_cmd'])
            self.ts_cmd=np.array(dict['ts_cmd'])
            self.alpha=np.array(dict['alpha'])
            self.cmd=np.array(dict['cmd'])
        else:
            self.epochs = dict['epoch']
            self.rewards = dict['reward']
            self.loss = dict['loss']


if __name__ == '__main__':
    parent_dir = os.getcwd()
    print('Welcome to PLOTTER!')
    print('The aim of this program is to generate real plots from the files "plot_dicts.pkl".')
    print('The current working directory is: ',parent_dir)
    will=''
    while not will=='q':
        will=input('Please indicate which trial to analize (eg.prove-run1) or write "q":\n -> ')
        if will=='q':
            break
        dir=os.path.join(parent_dir,will)
        print('Looking into the folder: ',dir)
        if os.path.isdir(dir):
            get_plots(dir)
        else:
            print(f'There is no folder named "{will}" in {parent_dir}')

