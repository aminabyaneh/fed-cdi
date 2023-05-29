import nauka
import numpy as np
import os, sys, time, pdb
import torch

from   causal.models import CategoricalWorld



class Experiment:
    def __init__(self, a):
        self.a = type(a)(**a.__dict__)
        self.a.__dict__.pop("__argp__", None)
        self.a.__dict__.pop("__argv__", None)
        self.a.__dict__.pop("__cls__",  None)
    
    def run(self):
        """
        Dump observational and interventional samples from CategoricalWorld
        """
        ### Create world from graph file
        world = CategoricalWorld(self.a)
        
        ### ---------------------------------
        ### Sampling - Observational Data
        ### ---------------------------------
        smpiter = world.sampleiter(self.a.num_samples_obs)
        samplesObs = next(smpiter)
        samplesObs = samplesObs.transpose(0,1)
        
        ### ---------------------------------
        ### Sampling - Inteventional Data
        ### ---------------------------------
        samplesInt = []
        expSetting = []
        for node in range(self.a.num_vars):
            
            # Create intervention
            with world.intervention(node = [node]) as intervention:
                
                # Sample interventional data
                smpiter = world.sampleiter(self.a.num_samples_int)
                samples = next(smpiter)
                samplesInt.append(samples)
                
                # Create intervention setting list (experiment identifier)
                exp = torch.tensor([node])
                exp = exp.repeat_interleave(self.a.num_samples_int)
                expSetting.append(exp)
           
        # Reshape Samples 
        samplesInt = torch.stack(samplesInt)
        samplesInt = samplesInt.permute(0,2,1).reshape(-1, self.a.num_vars)
        
        # Reshape intervention settings and save them
        expSetting = torch.stack(expSetting)
        expSetting = expSetting.reshape(-1, 1)

        ### ----------------------------------------------------------
        print("Generated Data:")
        print(" - Observational Data:", samplesObs.shape)
        print(" - Interventional Data:", samplesInt.shape)
   
        ### ----------------------------------------------------------
        ### Save data as csv files
        ###  @A: --> change here  if you like some other output type
        ### ----------------------------------------------------------
    
        # Retrieve ground-truth DAG
        dag = world.gammagt.byte()
        
        # Store data
        os.makedirs(self.a.dump_dir, exist_ok=True)
        np.savetxt(os.path.join(self.a.dump_dir, "dag.csv"),            dag.numpy(),     fmt="%d", delimiter=",")
        np.savetxt(os.path.join(self.a.dump_dir, "dataObs.csv"),        samples.numpy(), fmt="%d", delimiter=",")
        np.savetxt(os.path.join(self.a.dump_dir, "dataInt.csv"),        samplesInt.numpy(), fmt="%d", delimiter=",")
        np.savetxt(os.path.join(self.a.dump_dir, "dataInt_expId.csv"),  expSetting.numpy(), fmt="%d", delimiter=",")

