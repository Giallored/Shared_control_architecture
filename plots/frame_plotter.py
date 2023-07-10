import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv



def get_plots(parent_dir):
    check_plot=False
    while not check_plot:
        folder_name=input('What subforlder?: ')
        folder_name='testepoch_3'
        check_plot = elegibility("plot_dict.pkl",parent_dir,folder_name)
    plot = Plot(parent_dir,folder_name,"plot_dict.pkl")
    plot.save_frames()
    plot.close()
    
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


class Plot():
    def __init__(self,parent_dir,name,dict_name):
        self.parent_dir = parent_dir
        self.name=name
        self.dir = os.path.join(parent_dir,self.name)

        self.load_dict(dict_name)

    def save_frames(self,show=True):
        pc_0 =self.obs[0] 
        img_0 = pc2img(pc_0)
        #print(img_0.dtype)
        img_array = [img_0, img_0,img_0]
        new_img = np.concatenate(img_array, axis=0)
        #fig = plt.figure()
        #plt.imshow(new_img,cmap='gray', vmin=0, vmax=255,origin='lower')
        #plt.show()
        for step in self.obs.keys():
            pc =self.obs[step] 
            img = pc2img(pc)
            img_array.pop()
            img_array.append(img)
            new_img = np.concatenate(img_array, axis=0)
            fig = plt.figure()
            plt.imshow(new_img,cmap='gray', vmin=0, vmax=255,origin='lower')
            plt.show()
    

        #path = os.path.join(self.dir,str(step)+'_frame.png')
        #plt.savefig(path)
        #plt.close('all')
#

    

    def close(self):
        plt.close('all')

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
        self.obs = dict['obs']
        


def pc2img(pc,defi=2,height=300,width=400):
    pc = np.around(pc,decimals=defi)*10**defi#clean
    pc[:,1]+=width/2      #translate
    #print((pc[:,0]>=0) & (pc[:,0]<=300)&(pc[:,1]>=0) & (pc[:,1]<=height))
    pc = pc[(pc[:,0]>0)  & (pc[:,0]<height)  & (pc[:,1]>0)  & (pc[:,1]<width)] #crop
    pc=np.array(pc).astype('int') 
    rows = pc[:,0]
    cols = pc[:,1]
    img = np.zeros((height,width))
    img[rows,cols]=255
    kernel = np.ones((5,5),np.uint8)#
    img = cv.dilate(img,kernel,iterations = 1)
    #img = cv.resize(img, (400,300), interpolation = cv.INTER_AREA)
    return img


if __name__ == '__main__':
    parent_dir = os.getcwd()
    print('Welcome to PROVA!')
    print('The aim of this program is to generate real plots from the files "plot_dicts.pkl".')
    print('The current working directory is: ',parent_dir)
    will=''
    while not will=='q':
        will=input('Please indicate which trial to analize (eg.prove-run1) or write "q":\n -> ')
        if will=='q':
            break
        will='static-run85'
        dir=os.path.join(parent_dir,will)
        print('Looking into the folder: ',dir)
        if os.path.isdir(dir):
            get_plots(dir)
        else:
            print(f'There is no folder named "{will}" in {parent_dir}')

