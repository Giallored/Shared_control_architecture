import os
import pickle
import matplotlib.pyplot as plt


def get_plots(parent_dir):
    for folder_name in os.listdir(parent_dir):
        check_plot = elegibility("plot_dict.pkl",parent_dir,folder_name)
        check_frame = elegibility("frames.pkl",parent_dir,folder_name)
        if check_frame:
            print(f' - {folder_name} -> YES')
            plot = Plot(parent_dir,folder_name,"plot_dict.pkl")
            print('pippo')
            plot.save_plot()
            plot.close()
            
            frame = Frame(parent_dir,folder_name,"frames.pkl")
            frame.save_plot()
        else:
            print(f' - {folder_name} -> NO')

def elegibility(name,parent_dir,folder_name):
    folder = os.path.join(parent_dir,folder_name)
    path_pkl = os.path.join(folder,name) 
    list 
    if not os.path.exists(path_pkl): 
        return False
    else:
        list_file = os.listdir(folder)
        for file in list_file:
            if file[-4:]=='.png':
                return False
    return True

class Frame():
    def __init__(self,parent_dir,name,dict_name):
        self.parent_dir = parent_dir
        self.name=name
        self.dir = os.path.join(parent_dir,self.name)

        self.load_dict(dict_name)

    def load_dict(self,dict_name):
        where = os.path.join(self.dir,dict_name)
        with open(where, 'rb') as handle:
            self.frame_dict = pickle.load(handle)
    
    def save_plot(self):
        print('here')
        for i in range(1,10):
            try:
                frame = self.frame_dict[i]
                plt.plot(frame['poin_cloud'][:,0],frame['poin_cloud'][:,1],'y.')
                plt.plot(frame['cluster'][:,0],frame['cluster'][:,1],'b.')
                plt.plot(frame['X_obs'][0],frame['X_obs'][1],'r.')
                plt.plot(frame['centroid'][0],frame['centroid'][1],'g.')
                path = os.path.join(self.dir,'frame_' + str(i) + '.png')
                plt.savefig(path)
            except:
                pass
        


class Plot():
    def __init__(self,parent_dir,name,dict_name):
        self.parent_dir = parent_dir
        self.name=name
        self.dir = os.path.join(parent_dir,self.name)

        self.load_dict(dict_name)

 
    

    
    def save_plot(self,show=True):
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

        f_a= plt.plot( self.timesteps,self.alpha)
        plt.legend(['usr_a','ca_a','ts_a'])
        path = os.path.join(self.dir,'alpha.png')
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

    def save_dict(self):
        dict = {
            'timesteps':self.timesteps,
            'usr_cmd':self.usr_cmd,
            'ca_cmd':self.ca_cmd,
            'ts_cmd':self.ts_cmd,
            'cmd':self.cmd,
            'alpha':self.alpha,
            'env':self.env,
            'goal':self.goal
        }
        where = os.path.join(self.dir,'plot_dict.pickle')
        with open(where, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dict(self,dict_name):
        where = os.path.join(self.dir,dict_name)
        with open(where, 'rb') as handle:
            dict = pickle.load(handle)

        self.timesteps=dict['timesteps']
        self.usr_cmd=dict['usr_cmd']
        self.ca_cmd=dict['ca_cmd']
        self.ts_cmd=dict['ts_cmd']
        self.alpha=dict['alpha']
        self.cmd=dict['cmd']
        self.goal = dict['goal']
        self.env = dict['env']
        



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

